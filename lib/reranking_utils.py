def get_reranking_hparam(d2p_run, p2d_run):
    match (d2p_run.config_class.dataset, p2d_run.config_class.architecture):
        # this is average from reranking_eval2
        case ('chinese_baxter', 'GRU'):
            return round(7.2), 1.7550000000000003
        case ('chinese_baxter', 'Transformer'):
            return round(6.50000001), 2.43
        case ('chinese_baxter', 'JambuTransformer'):
            return round(6.50000001), 2.415
        case ('chinese_wikihan2022', 'GRU'):
            return round(5.9), 1.395
        case ('chinese_wikihan2022', 'Transformer'):
            return round(6.3), 1.275
        case ('chinese_wikihan2022', 'JambuTransformer'):
            return round(6.6), 1.2600000000000002
        case ('chinese_wikihan2022_augmented', 'GRU'):
            return round(7.0), 1.6199999999999997
        case ('chinese_wikihan2022_augmented', 'Transformer'):
            return round(8.3), 1.7550000000000001
        case ('chinese_wikihan2022_augmented', 'JambuTransformer'):
            return round(6.6), 1.5749999999999997
        case ('Nromance_ipa', 'GRU'):
            return round(5.4), 0.41999999999999993
        case ('Nromance_ipa', 'Transformer'):
            return round(6.1), 0.5549999999999999
        case ('Nromance_ipa', 'JambuTransformer'):
            return round(6.0), 0.5850000000000001
        case ('Nromance_orto', 'GRU'):
            return round(6.1), 0.8699999999999999
        case ('Nromance_orto', 'Transformer'):
            return round(5.5000001), 0.99
        case ('Nromance_orto', 'JambuTransformer'):
            return round(5.2), 0.915
        case _:
            raise NotImplemented