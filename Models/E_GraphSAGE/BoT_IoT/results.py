report = {
    # Host
    'host': {
        1500: { # DONE
            "DDoS": {
                "out_degree": (1, 412.98, 1500),
                "in_degree": (1, 1419.12, 1500),
                "influence": (0.0, 0.08, 1500),
                "normal_f1": 1.00,
                "to_both_f1": 0.79,
                "to_src_f1": 0.81,
                "to_dst_f1": 0.99,
                "edge_perturb_f1": 0.99,
                "random_edge_f1": 0.86
            },
            "DoS": {
                "out_degree": (1, 440.29, 1500),
                "in_degree": (1, 440.29, 1500),
                "influence": (0.0, 0.29, 1500),
                "normal_f1": 0.99,
                "to_both_f1": 0.57,
                "to_src_f1": 0.64,
                "to_dst_f1": 0.99,
                "edge_perturb_f1": 0.98,
                "random_edge_f1": 0.79
            },
            "Reconnaissance": {
                "out_degree": (1, 462.73, 1087),
                "in_degree": (1, 481.24, 1500),
                "influence": (0.0, 0.30, 787.71),
                "normal_f1": 0.88,
                "to_both_f1": 0.48,
                "to_src_f1": 0.37,
                "to_dst_f1": 0.88,
                "edge_perturb_f1": 0.88,
                "random_edge_f1": 0.25
            },
            "Weighted Average": {
                "influence": (0.0, 0.0, 0.0),
                "normal_f1": 0.99,
                "to_both_f1": 0.68,
                "to_src_f1": 0.72,
                "to_dst_f1": 0.99,
                "edge_perturb_f1": 0.98,
                "random_edge_f1": 0.81
            }
        },
        1000: { # DONE
            "DDoS": {
                "out_degree": (1, 286.84, 1000),
                "in_degree": (1, 970.90, 1000),
                "influence": (0.0, 0.08, 1000),
                "normal_f1": 1.00,
                "to_both_f1": 0.86,
                "to_src_f1": 0.88,
                "to_dst_f1": 0.99,
                "edge_perturb_f1": 0.99,
                "random_edge_f1": 0.70
            },
            "DoS": {
                "out_degree": (1, 298.80, 1000),
                "in_degree": (1, 298.80, 1000),
                "influence": (0.0, 0.30, 1000),
                "normal_f1": 1.00,
                "to_both_f1": 0.77,
                "to_src_f1": 0.80,
                "to_dst_f1": 0.99,
                "edge_perturb_f1": 0.99,
                "random_edge_f1": 0.03
            },
            "Reconnaissance": {
                "out_degree": (1, 248.79, 906),
                "in_degree": (1, 324.00, 1000),
                "influence": (0.0, 0.19, 410.42),
                "normal_f1": 1.00,
                "to_both_f1": 1.00,
                "to_src_f1": 0.97,
                "to_dst_f1": 1.00,
                "edge_perturb_f1": 1.00,
                "random_edge_f1": 0.88          
            },
            "Weighted Average": {
                "influence": (0.0, 0.0, 0.0),
                "normal_f1": 1.00,
                "to_both_f1": 0.82,
                "to_src_f1": 0.84,
                "to_dst_f1": 0.99,
                "edge_perturb_f1": 0.99,
                "random_edge_f1": 0.41   
            }
        },
        500: { # DONE
            "DDoS": {
                "out_degree": (1, 168.02, 500),
                "in_degree": (1, 483.28, 500),
                "influence": (0.0, 0.12, 500),
                "normal_f1": 0.99,
                "to_both_f1": 0.70,
                "to_src_f1": 0.66,
                "to_dst_f1": 0.98,
                "edge_perturb_f1": 0.99,
                "random_edge_f1": 0.72   
            },
            "DoS": {
                "out_degree": (1, 174.17, 500),
                "in_degree": (1, 174.17, 500),
                "influence": (0.0, 0.35, 500),
                "normal_f1": 0.99,
                "to_both_f1": 0.78,
                "to_src_f1": 0.77,
                "to_dst_f1": 0.97,
                "edge_perturb_f1": 0.98,
                "random_edge_f1": 0.23  
            },
            "Reconnaissance": {
                "out_degree": (1, 141.06, 500),
                "in_degree": (1, 181.89, 500),
                "influence": (0.0, 0.22, 500),
                "normal_f1": 0.95,
                "to_both_f1": 0.78,
                "to_src_f1": 0.78,
                "to_dst_f1": 0.95,
                "edge_perturb_f1": 0.95,
                "random_edge_f1": 0.61             
            },
            "Weighted Average": {
                "influence": (0.0, 0.0, 0.0),
                "normal_f1": 0.99,
                "to_both_f1": 0.74,
                "to_src_f1": 0.71,
                "to_dst_f1": 0.97,
                "edge_perturb_f1": 0.98,
                "random_edge_f1": 0.50   
            }
        },
    },
    'endpoint': {
        1500: { # DONE
            "DDoS": {
                "out_degree": (1, 1.00, 197),
                "in_degree": (1, 363, 1500),
                "influence": (0.0, 0.00, 25.87),
                "normal_f1": 1.00,
                "to_both_f1": 0.99,
                "to_src_f1": 0.97,
                "to_dst_f1": 1.00,
                "edge_perturb_f1": 0.99,
                "random_edge_f1": 0.97
            },
            "DoS": {
                "out_degree": (1, 1.00, 1),
                "in_degree": (1, 440.29, 1500),
                "influence": (0.0, 0.00, 0.00),
                "normal_f1": 1.00,
                "to_both_f1": 0.96,
                "to_src_f1": 0.95,
                "to_dst_f1": 1.00,
                "edge_perturb_f1": 0.99,
                "random_edge_f1": 0.95
                
            },
            "Reconnaissance": {
                "out_degree": (1, 2.77, 1023),
                "in_degree": (1, 1.60, 1500),
                "influence": (0.0, 0.00, 697.69),
                "normal_f1": 0.99,
                "to_both_f1": 0.65,
                "to_src_f1": 0.73,
                "to_dst_f1": 0.99,
                "edge_perturb_f1": 0.66,
                "random_edge_f1": 0.79       
            },
            "Weighted Average": {
                "influence": (0.0, 0.0, 0.0),
                "normal_f1": 1.00,
                "to_both_f1": 0.97,
                "to_src_f1": 0.96,
                "to_dst_f1": 1.00,
                "edge_perturb_f1": 0.98,
                "random_edge_f1": 0.96
            }
        },
        1000: { # DONE
           "DDoS": {
                "out_degree": (1, 1.00, 119),
                "in_degree": (1, 688.84, 1000),
                "influence": (0.0, 0.00, 14.16),
                "normal_f1": 1.00,
                "to_both_f1": 0.97,
                "to_src_f1": 0.96,
                "to_dst_f1": 0.97,
                "edge_perturb_f1": 0.99,
                "random_edge_f1": 0.96
            },
            "DoS": {
                "out_degree": (1, 1.00, 34),
                "in_degree": (1, 287.37, 1000),
                "influence": (0.0, 0.00, 1.16),
                "normal_f1": 1.00,
                "to_both_f1": 0.97,
                "to_src_f1": 0.95,
                "to_dst_f1": 0.99,
                "edge_perturb_f1": 0.99,
                "random_edge_f1": 0.95
            },
            "Reconnaissance": {
                "out_degree": (1, 5.70, 906),
                "in_degree": (1, 1.42, 995),
                "influence": (0.0, 0.02, 820.84),
                "normal_f1": 1.00,
                "to_both_f1": 0.67,
                "to_src_f1": 0.57,
                "to_dst_f1": 0.62,
                "edge_perturb_f1": 0.82,
                "random_edge_f1": 0.56           
            },
            "Weighted Average": {
                "influence": (0.0, 0.0, 0.0),
                "normal_f1": 1.00,
                "to_both_f1": 0.96,
                "to_src_f1": 0.95,
                "to_dst_f1": 0.97,
                "edge_perturb_f1": 0.99,
                "random_edge_f1": 0.95
            }
        },
        500: { # DONE
            'DDoS': {
                'influence': (0.0, 0.0, 28.32),
                'out_degree': (1.0, 1.0, 119.0),
                'in_degree': (1.0, 185.97, 500.0),
                'normal_f1': 0.99,
                'to_both_f1': 0.98,
                'to_src_f1': 0.98,
                'to_dst_f1': 0.98,
                'edge_perturb_f1': 0.98,
                'random_edge_f1': 0.98
            },
            'DoS': {
                'influence': (0.0, 0.0, 73.73),
                'out_degree': (1.0, 1.0, 192.0),
                'in_degree': (1.0, 153.44, 500.0),
                'normal_f1': 0.98,
                'to_both_f1': 0.98,
                'to_src_f1': 0.97,
                'to_dst_f1': 0.97,
                'edge_perturb_f1': 0.98,
                'random_edge_f1': 0.97
            },
            'Reconnaissance': {
                'influence': (0.0, 0.02, 500.0),
                'out_degree': (1.0, 3.82, 500.0),
                'in_degree': (1.0, 1.37, 497.0),
                'normal_f1': 0.86,
                'to_both_f1': 0.86,
                'to_src_f1': 0.9,
                'to_dst_f1': 0.87,
                'edge_perturb_f1': 0.79,
                'random_edge_f1': 0.87
            },
            'Weighted Average': {
                'normal_f1': 0.98,
                'to_both_f1': 0.97,
                'to_src_f1': 0.97,
                'to_dst_f1': 0.97,
                'edge_perturb_f1': 0.97,
                'random_edge_f1': 0.97
            }
        }
    },
}