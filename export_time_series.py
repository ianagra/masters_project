import numpy as np
import pandas as pd

def export_time_series(data):
    # Importação dos dados
    df = pd.read_pickle('datasets/dados_' + data + '.pkl')

    # Filtros
    clients = df['ClientMac'].unique()
    sites = df['Site'].unique()

    med = []
    for c in clients:
        for j,s in enumerate(sites):
            df_d = df[(df.ClientMac == c) & (df.DownloadSite == s)]
            df_u = df[(df.ClientMac == c) & (df.UploadSite == s)]
                    
            if len(df_d) >= 100:
                np.savetxt(f'datasets/ts_ndt/{c}_{s}_d_rttmean.txt', df_d[['DataHora', 'DownRTTMean']].values, fmt=['%s', '%.3f'], delimiter=',')
                np.savetxt(f'datasets/ts_ndt/{c}_{s}_d_throughput.txt', df_d[['DataHora', 'Download']].values, fmt=['%s', '%.3f'], delimiter=',')
                np.savetxt(f'datasets/ts_ndt/{c}_{s}_d_retrans.txt', df_d[['DataHora', 'DownloadRetrans']].values, fmt=['%s', '%.3f'], delimiter=',')
                np.savetxt(f'datasets/ts_ndt/{c}_{s}_u_rttmean.txt', df_u[['DataHora', 'UpRTTMean']].values, fmt=['%s', '%.3f'], delimiter=',')
                np.savetxt(f'datasets/ts_ndt/{c}_{s}_u_throughput.txt', df_u[['DataHora', 'Upload']].values, fmt=['%s', '%.3f'], delimiter=',')
        
                inicio_d = df_d['DataHora'].iloc[0]
                fim_d = df_d['DataHora'].iloc[-1]
                num_med_d = len(df_d)
                mean_t_d = np.round(df_d['DataHora'].diff().mean().seconds/3600,1)
                num_med_u = len(df_u)
                file_prefix = f"{c}_{s}_"
                
                quant = {"client": c, 
                        "site": s, 
                        "inicio": inicio_d,
                        "fim": fim_d,
                        "num_med_d": num_med_d,
                        "num_med_u": num_med_u,
                        "mean_t": mean_t_d,
                        "file_prefix": file_prefix
                        }
                med.append(quant)
                
                # Verifica se não há número de medições de download-upload diferentes
                if num_med_d != num_med_u:
                    print(f'Client: {c}, Site: {s}, num_med_d:{num_med_d}, num_med_u:{num_med_u}')

    # Conjunto de medições
    df_series = pd.DataFrame(med)
    df_series.to_pickle('datasets/ts_' + data + '.pkl')

    return df_series