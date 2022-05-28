import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--name', type=str, default='BERT-CAST')
    parser.add_argument('--attn_type', type=str, default='TB')
    parser.add_argument('--bert_name', type=str, default='prajjwal1/bert-tiny',
                        choices=['bert-base-uncased', 'albert-base-v2', 'prajjwal1/bert-small', 'prajjwal1/bert-mini'])
    parser.add_argument('--dataset', type=str, default='MIND/small')

    parser.add_argument('--read_text', action='store_false')
    parser.add_argument("--npratio", type=int, default=4)
    parser.add_argument('--pretrain', type=str, default='bert')
    parser.add_argument('--body_type', type=str, default='body')
    parser.add_argument('--scaling', type=str, default='yes')
    parser.add_argument('--model_path', type=str, default='none')
    parser.add_argument('--eval', type=str, default='yes')  # train(), eval() 체크 parameter추가

    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--t_layer', type=int, default=8)
    parser.add_argument('--max_hist_len', type=int, default=50)
    parser.add_argument('--max_title_len', type=int, default=30)
    parser.add_argument('--max_body_len', type=int, default=80)
    parser.add_argument('--max_abstract_len', type=int, default=80)
    parser.add_argument('--vocab_size', type=int, default=30522)

    parser.add_argument('--negative_sample_num', type=int, default=4)
    parser.add_argument('--seed', type=int, default=-1, help='Seed for random number generator')
    parser.add_argument('--device_id', type=int, default=0, help='Device ID of GPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_dc', type=float, default=0.5, help='learning rate decay.')
    parser.add_argument('--lr_dc_step', type=int, default=5,
                        help='the number of steps after which the learning rate decay.')

    parser.add_argument('--weight_decay', type=float, default=0, help='Optimizer weight decay')

    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=3)

    parser.add_argument('--reg_term', type=float, default=0, help='Regularization Term (Lambda) ')  # reg_term

    parser.add_argument('--hidden_size', type=int, default=400, help='Transformation dimension of user encoder')
    parser.add_argument('--n_heads', type=int, default=15, help='Head number of multi-head self-attention')
    parser.add_argument('--n_dim', type=int, default=20, help='dimension of each head')
    parser.add_argument('--news_dim', type=int, default=64, help='news_dim')
    parser.add_argument('--pos_dim', type=int, default=64, help='pos_dim')

    parser.add_argument('--position_dim', type=int, default=300, help='Positional dimension of user encoder')
    parser.add_argument('--head_num', type=int, default=20, help='Head number of multi-head self-attention')

    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--attention_dim', type=int, default=200, help="Attention dimension")
    parser.add_argument('--word_embedding_dim', type=int, default=768, help='Word embedding dimension')
    parser.add_argument('--glove_dim', type=int, default=300, help='Word embedding dimension')

    parser.add_argument('--cnn_method', type=str, default='naive', choices=['naive', 'group3', 'group4', 'group5'],
                        help='CNN group')
    parser.add_argument('--cnn_kernel_num', type=int, default=400, help='Number of CNN kernel')
    parser.add_argument('--cnn_window_size', type=int, default=3, help='Window size of CNN kernel')

    parser.add_argument('--use_category', action='store_false')
    parser.add_argument('--category_dim', type=int, default=50, help='Category embedding dimension')
    parser.add_argument('--subcategory_dim', type=int, default=50, help='SubCategory embedding dimension')
    parser.add_argument('--category_num', type=int, default=50, help='Category embedding num')
    parser.add_argument('--subcategory_num', type=int, default=50, help='SubCategory embedding num')

    args = parser.parse_args()

    logging.info(args)
    return args


# main
if __name__ == "__main__":
    args = parse_args()
