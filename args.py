import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Cluster CAM arguments")
    
    # Model arguments
    parser.add_argument(
        '--mode',
        choices=['single', 'batch'],
        default='batch',
        help="Chế độ chạy: 'single' hay 'batch'"
    )
    
    parser.add_argument(
        '--model',
        choices=[
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'alzheimer_resnet18', "vgg16", "inception_v3"
        ],
        required=True,
        help='Chọn model để test'
    )

    
    parser.add_argument(
        '--layer-name',
        default='layer4',
        help="Tên layer để tính CAM"
    )
    
    parser.add_argument(
        '--num-clusters',
        type=int,
        default=5,
        help="(chỉ dùng khi --cam-method=cluster) Số cluster cho ClusterScoreCAM"
    )
    
    parser.add_argument(
        '--img-path',
        type=str,
        help="(mode 'single') Đường dẫn tới ảnh"
    )
    
    parser.add_argument(
        '--save-prefix',
        type=str,
        default='output',
        help="(mode 'single') Prefix file heatmap"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help="(mode 'batch') Thư mục chứa ảnh hoặc file list"
    )
    
    parser.add_argument(
        '--excel-path',
        type=str,
        help="(mode 'batch') File hoặc thư mục Excel lưu kết quả"
    )
    
    parser.add_argument(
        '--k-values',
        nargs='+',
        type=int,
        default=[5],
        help="(mode 'batch', chỉ dùng khi --cam-method=cluster) Các giá trị K thử"
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=None,
        help="(mode 'batch') Số ảnh đầu tiên để test (mặc định None ⇒ load tất cả)"
    )
    
    parser.add_argument(
        '--cam-method',
        type=str,
        choices=[
            "cluster", "gradcam", "gradcamplusplus",
            "layercam", "scorecam", "ablationcam", "shapleycam"
        ],
        default="cluster",
        help="Phương pháp CAM để sử dụng"
    )
    
    parser.add_argument(
        "--zero-ratio",
        type=float,
        default=0.5,
        help="Tỷ lệ zero trong ClusterScoreCAM"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature cho softmax trong ClusterScoreCAM"
    )

    args = parser.parse_args()

    # Validate
    if args.mode == 'single':
        if not args.img_path:
            parser.error("--img-path là bắt buộc khi mode='single'")
    else:  # batch
        if not args.dataset or not args.excel_path:
            parser.error("--dataset và --excel-path là bắt buộc khi mode='batch'")

    return args
