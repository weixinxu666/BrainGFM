import pandas as pd


def xlsx_to_csv_pd():
    # data_xls = pd.read_excel('/home/xinxu/Lehigh/Codes/Lehigh_graph/xxw_lehigh/data/ADNI_gene/ADNI_sMRI_PET_Genetics_LiShen/[02]-imaging/AV45/AV45-MRI.xlsx', index_col=0)
    # data_xls.to_csv('/home/xinxu/Lehigh/Codes/Lehigh_graph/xxw_lehigh/data/ADNI_gene/ADNI_sMRI_PET_Genetics_LiShen/[02]-imaging/AV45/AV45-MRI.csv', encoding='utf-8')

    # data_xls = pd.read_excel('/home/xinxu/Lehigh/Codes/Lehigh_graph/xxw_lehigh/data/ADNI_gene/ADNI_sMRI_PET_Genetics_LiShen/[02]-imaging/FDG/FDG-PET.xlsx', index_col=0)
    # data_xls.to_csv('/home/xinxu/Lehigh/Codes/Lehigh_graph/xxw_lehigh/data/ADNI_gene/ADNI_sMRI_PET_Genetics_LiShen/[02]-imaging/FDG/FDG-PET.csv', encoding='utf-8')

    data_xls = pd.read_excel('/home/xinxu/Lehigh/Codes/BICLab_data/ADNI/ADNIMERGE.xlsx', index_col=0)
    data_xls.to_csv('/home/xinxu/Lehigh/Codes/BICLab_data/ADNI/ADNIMERGE.csv', encoding='utf-8')


if __name__ == '__main__':
    xlsx_to_csv_pd()
