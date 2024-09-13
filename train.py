
if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='LHRL')  # LHRLCNN
    parser.add_argument('--env_mode', type=str, default='left_t')  # or merge
    parser.add_argument('--log_dir', type=str, default='./log/LHRL/')
    parser.add_argument('--encoder_model_path', type=str)
    parser.add_argument('--expert_data_name', type=str)
    args = parser.parse_args()

    if args.alg == 'LHRL':
        from algorithm.LikeHumanRL.trainer import trainer
        trainer(log_dir=args.log_dir, env_mode=args.env_mode, encoder_model_path=args.encoder_model_path, expert_data_name=args.expert_data_name)
