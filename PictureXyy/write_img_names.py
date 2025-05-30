import os

def write_img_names(folder, txt_name='aimg.txt'):
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    img_names = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in img_exts]
    with open(os.path.join(folder, txt_name), 'w', encoding='utf-8') as f:
        for name in img_names:
            f.write(name + '\n')
    print(f'已将{len(img_names)}个图片文件名写入 {txt_name}')

if __name__ == '__main__':
    folder = os.path.dirname(__file__)
    write_img_names(folder)
