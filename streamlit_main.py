import streamlit as st
import pandas as pd
from test_classification import test_main
from train_classification import train_main
import argparse
from PIL import Image

st.set_page_config(layout="wide")

def parse_args_test():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size in training')
    parser.add_argument('--num_category', default=4, type=int, choices=[4, 10, 40],  help='training on MyTensor')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='pointnet2_cls_ssg', help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--fileName', type=str, default='Tensor0_000020', help='the filename to test')
    return parser.parse_args()

def parse_args_train():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=4, type=int, choices=[4,10, 40],  help='training on MyTensor')
    parser.add_argument('--epoch', default=20, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default='pointnet2_cls_ssg', help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampiling')
    return parser.parse_args()

def show_images(imgs, index=0):
    if(len(imgs)>0):
        st.image(imgs[index])

# if __name__ == '__main__':
#     args = parse_args_test()
#     args.fileName = 'Tensor2_000020'
#     test_main(args)

def ui_train():
    col_train_para, col_train_result = st.columns([1, 2], gap="medium")
    with col_train_para:
        st.text("训练参数：")
        batch_size = st.slider('batch Size', 1, 16, 2)
        epoch = st.slider('epoch', 10, 200, 20)
        num_category_train = st.slider('分类种类数(训练)', 2, 40, 4)
        num_point_train = st.number_input('点云数量(训练)', 512, 4096, 1024, 32)
        use_normals_train = st.checkbox('使用法线(训练)', False)
        with st.form("image form", clear_on_submit=True):
            uploaded_files = st.file_uploader('查看训练图像', type=['png', 'jpeg', 'jpg'], accept_multiple_files=True)
            submitted = st.form_submit_button("清空所有图像")
            if uploaded_files is not None:
                train_imgs = []
                for uploaded_file in uploaded_files:
                    img_file_path = 'data/mytensor_normal_data/Test/' + uploaded_file.name
                    img = Image.open(img_file_path)
                    train_imgs.append(img)
                if (len(train_imgs) > 0):
                    imgs_index = st.slider('图像索引', 0, len(train_imgs) - 1, 0)
                    show_images(train_imgs, imgs_index)

    with col_train_result:
        st.text("训练结果：")
        args_train = parse_args_train()
        for_test = st.checkbox('测试')
        if st.button('开始训练'):
            args_train.num_category = num_category_train
            args_train.num_point = num_point_train
            args_train.use_normals = use_normals_train
            args_train.batch_size = batch_size
            args_train.epoch = epoch

            with st.spinner('正在训练，请稍后 ...'):
                if for_test:
                    train_instance_acc_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                else:
                    train_instance_acc_list = train_main(args_train)
            st.success('训练完成!')
            chart_data = pd.DataFrame(train_instance_acc_list, columns=['训练精度'])
            st.line_chart(chart_data)

def ui_test():
    img_file = None
    col_test_para, col_test_result = st.columns([1, 2], gap="medium")
    with col_test_para:
        st.text("测试参数：")
        num_category = st.slider('分类种类数', 2, 40, 4)
        num_point = st.number_input('点云数量', 512, 4096, 1024, 32)
        use_normals = st.checkbox('使用法线', False)
        args = parse_args_test()
        img_file = st.file_uploader("请上传测试图像", type=['png', 'jpg', 'jpeg'])
        if img_file is not None:
            with st.spinner(text='图像加载中，请稍后'):
                st.image(img_file)
                img_file_name = img_file.name.split('.')[0]
                st.write(f'fileName = {img_file_name}')
                args.fileName = img_file_name

    with col_test_result:
        st.text("测试结果：")
        if st.button("开始预测"):
            args.num_category = num_category
            args.num_point = num_point
            args.use_normals = use_normals
            with st.spinner('正在预测，请稍后 ...'):
                cls_result, spend_time = test_main(args)
            st.image(img_file)
            st.success(f'预测结果[ 本点云的张力为：{cls_result}，消耗的时间为：{spend_time} ms ]', icon="✅")

col_logo,col_title = st.columns([1,10])
col_logo.image('./imgs/logo3.png')
col_title.title("智能视觉张力检测系统")
train, test = st.tabs(["训练","测试"])

with train:
    ui_train()

with test:
    ui_test()

