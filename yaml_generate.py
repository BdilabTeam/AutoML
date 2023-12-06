import os.path

import ruamel.yaml

dir = './jobs/'
def generate(job_name_list):
    for job_name in job_name_list:
        key = job_name.split('-')
        template_path = dir + key[0] + '.yaml'
        new_path = dir + job_name + '.yaml'

        if os.path.exists(new_path):
            continue

        yaml = ruamel.yaml.YAML()
        yaml.preserve_quotes = True  # 保留原始文件中的引号
        with open(template_path, 'r', encoding='utf-8') as f:
            data = yaml.load(f.read())
        data_new = data
        data_new['metadata']['name'] = job_name
        data_new['spec']['tfReplicaSpecs']['Worker']['replicas'] = int(key[2])
        args = data['spec']['tfReplicaSpecs']['Worker']['template']['spec']['containers'][0]['args']
        new_args = args
        new_args[1] = key[4]
        new_args[5] = key[3]
        if key[0] == 'unet':
            new_args[9] = job_name
        else:
            new_args[7] = job_name
        data_new['spec']['tfReplicaSpecs']['Worker']['template']['spec']['containers'][0]['args'] = new_args
        with open(new_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_new, f)


if __name__ == '__main__':
    job_name_list = ['resnet-2-2-10-16','resnet-2-5-20-16','rnn-5-2-10-16','rnn-5-5-30-8']
    generate(job_name_list)


