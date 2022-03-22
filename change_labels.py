import glob
import tqdm
def change_labels():
    listdir = glob.glob('./my_data/labels/*/*.txt')
    pbar = tqdm.tqdm(listdir)
    for dir in pbar:
        with open(dir) as rf:
            contents = rf.readlines()
            for i,content in enumerate(contents):
                content = list(content)
                content[0] = '0'
                content = ''.join(content)
                contents[i] = content
        with open(dir,'w') as wf:
            wf.write(''.join(contents))




if __name__ == '__main__':
    change_labels()