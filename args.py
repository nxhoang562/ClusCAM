import argparse 

def get_args():
    parser = argparse.ArgumentParser(description="Cluster CAM arguments")
    
    # Model arguments
    parser.add_argument('--mode', choices=['single', 'batch'], default='batch',
                        help="Chế độ chạy: 'single' hay 'batch'")
    
    parser.add_argument('--model', choices=['resnet18'], default='resnet18',
                        help="Chọn kiến trúc model")
    
    parser.add_argument('--layer-name', default='layer4',
                        help="Tên layer để tính CAM")
    
    parser.add_argument('--num-clusters', type=int, default=[5],
                        help="Số cluster cho ClusterScoreCAM")
    
    parser.add_argument('--img-path', type=str,
                        help="(mode 'single') Đường dẫn tới ảnh")
    
    parser.add_argument('--save-prefix', type=str, default='output',
                        help="(mode 'single') Prefix file heatmap")
    
    parser.add_argument('--dataset', type=str,
                        help="(mode 'batch') Thư mục chứa ảnh")
    
    parser.add_argument('--excel-path', type=str,
                        help="(mode 'batch') File Excel lưu kết quả")
    
    parser.add_argument('--k-values', nargs='+', type=int, default=[5],
                        help="(mode 'batch') Các giá trị K thử")
    
    parser.add_argument('--top-n', type=int, default=100,
                        help="(mode 'batch') Số ảnh đầu tiên để test")
    parser.add_argument('--cam', choices=['ClusterScoreCAM', 'ClusterScoreCAM2'], default='ClusterScoreCAM',
                        help="Chọn loại CAM: ClusterScoreCAM hay ClusterScoreCAM2")
    
    parser.add_argument("--cam-method",
                        type = str,
                        choices = ["cluster", "gradcam", "gradcamplusplus", "layercam", "scorecam", "ablationcam", "shapleycam"],
                        default = "cluster",
                        help = "Phương pháp CAM để sử dụng")
    
    parser.add_argument("--zero-ratio",
        type=float,
        default=0.5,
        help="Tỷ lệ zero trong ClusterScoreCAM",)
    
    parser.add_argument("--temperature",
                        type=float,
                        default=1.0,
                        help="temperature cho softmax trong ClusterScoreCAM",)

    return parser.parse_args()
    
    
