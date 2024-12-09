thresholding_mrns = ['21207670', '21260966', '21270369', '21275957', '21276499', '21316038', '21353148','21354530','21375845','40265487']
kevs_mrns = ['21221312', '21238848', '21282741', '21419169', '40690080']
totalsegmentator_mrns = ['21272522', '21310908', '21343104', '41644655', '41688648']

combined_file_names = thresholding_mrns + kevs_mrns + totalsegmentator_mrns

file_key = {filename: f"UCLH-Cyst_{i+1:04}" for i, filename in enumerate(combined_file_names)}

print(file_key)