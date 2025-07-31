import scipy.io as scio
from tqdm import tqdm


mat_data = scio.loadmat('/home/xinxu/Lehigh/Codes/BICLab_data/ADNI/adni23_label.mat')


data = mat_data['data']

adni = data[0]

# roi_data = data['ROISignals_Schaefer100']
sex = data['gender']
sub_id = data['subjectID']
dx = data['DX_bl']


id_list = []
dx_list = []

for i in tqdm(range(len(sub_id))):
    try:
        print(dx[0])
        dx_i = dx[i][0][0]
        id_i = sub_id[i][0][0]
        id_list.append(id_i)
        dx_list.append(dx_i)
    except IndexError:
        print(id_i, 'not in')







