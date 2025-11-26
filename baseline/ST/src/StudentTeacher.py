import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import reduce, rearrange
from sklearn.metrics import roc_curve, auc
from torchvision import transforms
from .AnomalyNet import AnomalyNet
from .AnomalyResnet18 import AnomalyResnet18
from .utils4st import load_model, increment_mean_and_var

class StudentTeacherAnomalyDetector(torch.nn.Module):
    def __init__(self, device):
        super(StudentTeacherAnomalyDetector, self).__init__()
        self.device = device
        self.resnet18 = None
        self.teacher = None
        self.students = []
        self.patch_size = None
        self.n_students = 0
        self.calibration_params = None
        self.model_dir = ""
        self.dataset_name = ""

    def load(self, 
             patch_size=65, 
             n_students=3, 
             **kwargs):
        self.patch_size = patch_size
        self.n_students = n_students
        
        # Initialize ResNet18 Backbone
        self.resnet18 = AnomalyResnet18()
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])
        self.resnet18.eval().to(self.device)

        # Initialize Teacher
        self.teacher = AnomalyNet.create((self.patch_size, self.patch_size))
        self.teacher.to(self.device)

        # Initialize Students
        self.students = [AnomalyNet.create((self.patch_size, self.patch_size)) for _ in range(self.n_students)]
        for s in self.students:
            s.to(self.device)

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(os.path.join(self.model_dir, dataset_name), exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, 'tb')
        os.makedirs(self.tb_dir, exist_ok=True)

    def _load_models_from_disk(self):
        """Load pretrained weights for ResNet, Teacher, and Students if they exist."""
        # Load ResNet
        resnet_path = os.path.join(self.model_dir, self.dataset_name, 'resnet18.pt')
        if os.path.exists(resnet_path):
            print(f"Loading ResNet from {resnet_path}")
            # Re-init to load state dict correctly before slicing
            temp_resnet = AnomalyResnet18()
            load_model(temp_resnet, resnet_path)
            self.resnet18 = nn.Sequential(*list(temp_resnet.children())[:-2])
            self.resnet18.eval().to(self.device)
        else:
            print(f"ResNet model not found at {resnet_path}, using ImageNet pretrained weights.")

        # Load Teacher
        teacher_path = os.path.join(self.model_dir, self.dataset_name, f'teacher_{self.patch_size}_net.pt')
        if os.path.exists(teacher_path):
            print(f"Loading Teacher from {teacher_path}")
            load_model(self.teacher, teacher_path)
        else:
            print(f"Teacher model not found at {teacher_path}")
        
        # Load Students
        for i in range(self.n_students):
            student_path = os.path.join(self.model_dir, self.dataset_name, f'student_{self.patch_size}_net_{i}.pt')
            if os.path.exists(student_path):
                print(f"Loading Student {i} from {student_path}")
                load_model(self.students[i], student_path)
            else:
                print(f"Student {i} model not found at {student_path}")

    def train(self, training_data, test_data=None, teacher_epochs=1000, student_epochs=15, learning_rate=2e-4, weight_decay=1e-5): # type: ignore
        self._load_models_from_disk()

        # Train Teacher
        self._train_teacher(training_data, teacher_epochs, learning_rate, weight_decay)

        # Train Students
        self._train_students(training_data, student_epochs, learning_rate/2, weight_decay) # Students usually use lower LR

        if test_data:
            return self.test(test_data)
        return None

    def _distillation_loss(self, output, target):
        # dim: (batch, vector)
        err = torch.norm(output - target, dim=1)**2
        loss = torch.mean(err)
        return loss
    

    def _compactness_loss(self, output):
        _, n = output.size()
        avg = torch.mean(output, axis=1)
        std = torch.std(output, axis=1)
        zt = output.T - avg
        zt /= std
        corr = torch.matmul(zt.T, zt) / (n - 1)
        loss = torch.sum(torch.triu(corr, diagonal=1)**2)
        return loss
    
    def _train_teacher(self, dataloader, max_epochs, learning_rate, weight_decay):
        print("Starting Teacher Training...")
        model_save_path = os.path.join(self.model_dir, self.dataset_name, f'teacher_{self.patch_size}_net.pt')
        
        optimizer = optim.Adam(self.teacher.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)
        
        min_running_loss = np.inf
        for epoch in range(max_epochs):
            running_loss = 0.0

            for i, batch in tqdm(enumerate(dataloader), desc=f"Teacher Epoch {epoch+1}/{max_epochs}"):
                optimizer.zero_grad()

                inputs = batch['image'].to(self.device)
                with torch.no_grad():
                    targets = rearrange(self.resnet18(inputs), 'b vec h w -> b (vec h w)')
                
                outputs = self.teacher(inputs)
                loss = self._distillation_loss(outputs, targets) + self._compactness_loss(outputs)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1} \t loss: {running_loss}")
            
            if running_loss < min_running_loss and epoch > 0:
                torch.save(self.teacher.state_dict(), model_save_path)
                print(f"Loss decreased: {min_running_loss} -> {running_loss}.")
                print(f"Model saved to {model_save_path}.")
                min_running_loss = running_loss
            
            running_loss = 0.0

    def _student_loss(self, output, target):
        # dim: (batch, h, w, vector)
        err = reduce((output - target)**2, 'b h w vec -> b h w', 'sum')
        loss = torch.mean(err)
        return loss
    
    def _train_students(self, dataloader, max_epochs, learning_rate, weight_decay):
        print("Starting Students Training...")
        
        # Preprocessing: Compute incremental mean and var over training set using Teacher
        print(f'Preprocessing of training dataset {self.dataset_name}...')
        self.teacher.eval()
        t_mu, t_var, N = 0, 0, 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), desc="Preprocessing"):
                inputs = batch['image'].to(self.device)
                t_out = self.teacher.fdfe(inputs)
                t_mu, t_var, N = increment_mean_and_var(t_mu, t_var, N, t_out)
        
        optimizers = [optim.Adam(student.parameters(), 
                                lr=learning_rate, 
                                weight_decay=weight_decay) for student in self.students]

        for j, student in enumerate(self.students):
            min_running_loss = np.inf
            model_name = os.path.join(self.model_dir, self.dataset_name, f'student_{self.patch_size}_net_{j}.pt')
            print(f'Training Student {j} on anomaly-free dataset ...')

            for epoch in range(max_epochs):
                running_loss = 0.0

                for i, batch in tqdm(enumerate(dataloader), desc=f"Student {j} Epoch {epoch+1}/{max_epochs}"):
                    optimizers[j].zero_grad()

                    inputs = batch['image'].to(self.device)
                    with torch.no_grad():
                        targets = (self.teacher.fdfe(inputs) - t_mu) / torch.sqrt(t_var)
                    
                    outputs = student.fdfe(inputs)
                    loss = self._student_loss(targets, outputs)

                    loss.backward()
                    optimizers[j].step()
                    running_loss += loss.item()

                print(f"Epoch {epoch+1} \t loss: {running_loss}")
                
                if running_loss < min_running_loss and epoch > 0:
                    torch.save(student.state_dict(), model_name)
                    print(f"Loss decreased: {min_running_loss} -> {running_loss}.")
                    print(f"Model saved to {model_name}.")
                    min_running_loss = running_loss

    def _get_error_map(self, students_pred, teacher_pred):
        # student: (batch, student_id, h, w, vector)
        # teacher: (batch, h, w, vector)
        mu_students = reduce(students_pred, 'b id h w vec -> b h w vec', 'mean')
        err = reduce((mu_students - teacher_pred)**2, 'b h w vec -> b h w', 'sum')
        return err

    def _get_variance_map(self, students_pred):
        # student: (batch, student_id, h, w, vector)
        sse = reduce(students_pred**2, 'b id h w vec -> b id h w', 'sum')
        msse = reduce(sse, 'b id h w -> b h w', 'mean')
        mu_students = reduce(students_pred, 'b id h w vec -> b h w vec', 'mean')
        var = msse - reduce(mu_students**2, 'b h w vec -> b h w', 'sum')
        return var

    @torch.no_grad()
    def calibrate(self, dataloader):
        print('Calibrating teacher on Student dataset.')
        self.teacher.eval()
        for s in self.students:
            s.eval()

        t_mu, t_var, t_N = 0, 0, 0
        for _, batch in tqdm(enumerate(dataloader), desc="Calibrating Teacher"):
            inputs = batch['image'].to(self.device)
            t_out = self.teacher.fdfe(inputs)
            t_mu, t_var, t_N = increment_mean_and_var(t_mu, t_var, t_N, t_out)
        
        print('Calibrating scoring parameters on Student dataset.')
        max_err, max_var = 0, 0
        mu_err, var_err, N_err = 0, 0, 0
        mu_var, var_var, N_var = 0, 0, 0

        for _, batch in tqdm(enumerate(dataloader), desc="Calibrating Scoring"):
            inputs = batch['image'].to(self.device)

            t_out = (self.teacher.fdfe(inputs) - t_mu) / torch.sqrt(t_var)
            s_out = torch.stack([student.fdfe(inputs) for student in self.students], dim=1)

            s_err = self._get_error_map(s_out, t_out)
            s_var = self._get_variance_map(s_out)
            mu_err, var_err, N_err = increment_mean_and_var(mu_err, var_err, N_err, s_err)
            mu_var, var_var, N_var = increment_mean_and_var(mu_var, var_var, N_var, s_var)

            max_err = max(max_err, torch.max(s_err))
            max_var = max(max_var, torch.max(s_var))
        
        self.calibration_params = {
            "teacher": {"mu": t_mu, "var": t_var},
            "students": {"err": {"mu": mu_err, "var": var_err, "max": max_err},
                         "var": {"mu": mu_var, "var": var_var, "max": max_var}}
        }
        return self.calibration_params

    @torch.no_grad()
    def get_score_map(self, inputs):
        if self.calibration_params is None:
            raise ValueError("Model must be calibrated before getting score map.")
        
        params = self.calibration_params
        t_out = (self.teacher.fdfe(inputs) - params['teacher']['mu']) / torch.sqrt(params['teacher']['var'])
        s_out = torch.stack([student.fdfe(inputs) for student in self.students], dim=1)

        s_err = self._get_error_map(s_out, t_out)
        s_var = self._get_variance_map(s_out)
        score_map = (s_err - params['students']['err']['mu']) / torch.sqrt(params['students']['err']['var'])\
                        + (s_var - params['students']['var']['mu']) / torch.sqrt(params['students']['var']['var'])
        
        return score_map

    def visualize(self, img, gt, score_map, max_score):
        plt.figure(figsize=(13, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title(f'Original image')

        plt.subplot(1, 3, 2)
        plt.imshow(gt, cmap='gray')
        plt.title(f'Ground thuth anomaly')

        plt.subplot(1, 3, 3)
        plt.imshow(score_map, cmap='jet')
        plt.imshow(img, cmap='gray', interpolation='none')
        plt.imshow(score_map, cmap='jet', alpha=0.5, interpolation='none')
        plt.colorbar(extend='both')
        plt.title('Anomaly map')

        plt.clim(0, max_score)
        plt.show(block=True)

    def test(self, dataloader, test_size=20, visualize=False):
        print("Starting Testing/Anomaly Detection...")
        self.teacher.eval()
        for s in self.students:
            s.eval()

        y_score = np.array([])
        y_true = np.array([])
        test_iter = iter(dataloader)

        for i in range(test_size):
            try:
                batch = next(test_iter)
            except StopIteration:
                break
                
            inputs = batch['image'].to(self.device)
            gt = batch['gt'].cpu()

            score_map = self.get_score_map(inputs).cpu()
            y_score = np.concatenate((y_score, rearrange(score_map, 'b h w -> (b h w)').numpy()))
            y_true = np.concatenate((y_true, rearrange(gt, 'b c h w -> (b c h w)').numpy()))

            if visualize:
                params = self.calibration_params
                unorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))
                max_score = (params['students']['err']['max'] - params['students']['err']['mu']) / torch.sqrt(params['students']['err']['var'])\
                    + (params['students']['var']['max'] - params['students']['var']['mu']) / torch.sqrt(params['students']['var']['var']).item()
                img_in = rearrange(unorm(inputs).cpu(), 'b c h w -> b h w c')
                gt_in = rearrange(gt, 'b c h w -> b h w c')

                for b in range(inputs.size(0)):
                    self.visualize(img_in[b, :, :, :].squeeze(), 
                                   gt_in[b, :, :, :].squeeze(), 
                                   score_map[b, :, :].squeeze(), 
                                   max_score)
        
        # AUC ROC
        fpr, tpr, thresholds = roc_curve(y_true.astype(int), y_score)
        auc_score = auc(fpr, tpr)
        print(f'ROC AUC: {auc_score}')
        
        # plt.figure(figsize=(13, 3))
        # plt.plot(fpr, tpr, 'r', label="ROC")
        # plt.plot(fpr, fpr, 'b', label="random")
        # plt.title(f'ROC AUC: {auc_score}')
        # plt.xlabel('FPR')
        # plt.ylabel('TPR')
        # plt.legend()
        # plt.grid()
        # plt.show()
        
        return auc_score
