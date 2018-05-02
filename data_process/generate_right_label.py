label_txt_root = '../../data/tmp_data/label.txt'
right_label_txt_root = '../../data/tmp_data/right_label.txt'

fi = open(label_txt_root, 'r')
fo = open(right_label_txt_root, 'w')

for i in fi:
    i = i.strip().split('\t')
    fo.write(i[0] + '\t' + ' a' * 100 + '\n')

fi.close()
fo.close()