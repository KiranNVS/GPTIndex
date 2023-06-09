from datetime import datetime, timedelta
import re
import os.path


class ICEWSDataset:
    START_DATES = {
        'ICEWS05-15': '01/01/2005',
        'ICEWS14': '01/01/2014',
        'ICEWS18': '01/01/2018',
    }
    
    def __init__(self, dir_path, dataset_name, filename, idx):
        self.path = dir_path
        self.dataset_name = dataset_name

        self.ent2id = self.get_id_dic('entity2id')
        self.rel2id = self.get_id_dic('relation2id')
        self.data = self.get_facts_by_idx(filename, idx)  # train, valid, test

    def load_data(self, filename):
        if self.dataset_name not in ['ICEWS14', 'ICEWS05-15', 'ICEWS18']:
            raise ValueError('Unknown dataset: {}'.format(dataset))

        file = open(os.path.join(self.path, self.dataset_name, filename+'.txt'), 'r', encoding='utf-8')
        
        data = []
        for l in file.readlines():
            line = l.split()
            if len(line) > 4: # some data have 0 or -1 appended at the tail
                data.append(line[:4])
            else:
                data.append(line)
        return data

    def get_id_dic(self, filename):
        text = self.load_data(filename)
        
        dic = {}
        for l in text:
            dic[int(l[1])] = l[0]
        return dic

    def get_facts_by_idx(self, filename, idx):
        start_date = self.START_DATES[self.dataset_name]
        start_datetime = datetime.strptime(start_date, '%m/%d/%Y')
        text = self.load_data(filename)

        quadruples = []
        for l in text:
            sub_id = int(l[0])
            rel_id = int(l[1])
            obj_id = int(l[2])
            t_id = start_datetime + timedelta(days=int(l[3]) // 24)
            quadruples.append([self.ent2id[sub_id],
                               self.rel2id[rel_id],
                               self.ent2id[obj_id],
                               str(t_id.date())  # TODO: what to do with time -- sort?
                               ])
        
        if idx[0] < len(quadruples) and idx[1] < len(quadruples):
            return quadruples[idx[0]: idx[1]]
        else:
            raise IndexError("Indices of the data attempted to retrieve is out of range")


if __name__ == "__main__":
    dir_path = '../data'           # parent dir path of the datasets
    dataset_name = 'ICEWS14'  # 'ICEWS14', 'ICEWS05-15', 'ICEWS18'
    mode = 'train'            # train, valid, test
    idx = (1400,1415)           # index of facts to retrieve

    # usage
    dataset = ICEWSDataset(dir_path, dataset_name, mode, idx)
    item = dataset.data

    for i in item:
        pattern = r'[\[\]]'
        print(re.sub(pattern, "", str(i)))
