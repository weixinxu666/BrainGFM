import scipy.io as scio

# 读取MAT文件
# mat_data = scio.loadmat('/home/xinxu/Lehigh/Codes/Lehigh_graph/xxw_lehigh/data/ADHD200/ADHD200/ADHD200_rsfMRI_ROISignals_Schaefer100ROIs.mat')
# mat_data = scio.loadmat('/home/xinxu/Lehigh/Codes/Lehigh_graph/xxw_lehigh/data/ADNI/ADNI2.mat')
mat_data = scio.loadmat('/home/xinxu/Lehigh/Codes/BICLab_data/ADNI/adni23_label.mat')


data = mat_data['data']

adni = data[0]

roi_data = data['ROISignals_Schaefer100']
sex = data['gender']
sub_id = data['subjectID']
dx = data['DX_bl']

print(data['ROISignals_Schaefer100'])
print(data['subjectID'])
print(data['subjectID'])





