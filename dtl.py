import numpy as np
from tabulate import tabulate

def sample_data():
  hp = ["H" + str(i) for i in range(1, 15)]
  merek = ["M" + str(i) for i in range(1, 15)]
  layak = (['Y'] * 3) + ['T'] + (['Y'] * 4) + ['T'] + ['Y'] + (['T'] * 4)
  baterai = (['Kuat'] * 4) + (['Cukup'] * 5 + (['Lemah'] * 5))
  kamera = (['Tinggi'] * 2) + ['Sedang'] + ['Rendah'] + ['Tinggi'] + (['Sedang'] * 2) + ['Tinggi'] + ['Rendah'] + (['Tinggi'] * 2) + (['Sedang'] * 2) + ['Rendah']
  harga = (['Sangat Murah'] * 2) + (['Mahal'] * 2) + ['Sangat Murah'] + ['Mahal'] + ['Sangat Murah'] + (['Murah'] * 2) + (['Sangat Murah'] * 2) + (['Mahal'] * 2) + ['Sangat Mahal']
  return np.c_[hp, baterai, kamera, merek, harga, layak]

def index_in_list(l,i):
  return len(l) < i

"""
Group data
ex :
[['Y' 'Kuat']
 ['Y' 'Kuat']
 ['Y' 'Kuat']
 ['T' 'Kuat']
 ['Y' 'Cukup']
 ['Y' 'Cukup']
 ['Y' 'Cukup']
 ['Y' 'Cukup']
 ['T' 'Cukup']
 ['Y' 'Lemah']
 ['T' 'Lemah']
 ['T' 'Lemah']
 ['T' 'Lemah']
 ['T' 'Lemah']]
usage: 
groupFn(data, data[:,1], ret_col_idx=0)
return:
(['Cukup', 'Kuat', 'Lemah']) [['Y', 'Y', 'Y', 'Y', 'T'], ['Y', 'Y', 'Y', 'T'], ['Y', 'T', 'T', 'T', 'T']]
"""
def groupFn(data, pivot = None, ret_col_idx = None):
  def create_key(d):
    if not isinstance(d, list):
      if isinstance(d, np.ndarray):
        d = d.tolist()
      else:
        d = [d]
    return ".".join(d)
  if (pivot is None):
    pivot = data
  mapped = {}
  for i, d in enumerate(data):
    key = create_key(pivot[i])
    d = d[ret_col_idx] if ret_col_idx is not None else d
    if key not in mapped:
      mapped[key] = [d]
    else:
      mapped[key].append(d)
  return mapped.keys(), mapped.values()

"""
Hitung nilai entropy (heterogenitas) data
"""
def entropy(sample):
  total = len(sample)
  distinct, distinct_counts = np.unique(sample, return_counts=True)
  return sum(-(float(cnt/total)) * np.log2(float(cnt/total)) for cnt in distinct_counts)

def gain_ratio(sample, attribute):
  return gain(sample, attribute) / split_information(sample, attribute)

"""
Hitung nilai efektifitas attribut dalam data
"""
def gain(sample, attribute):
  cctd = np.c_[sample, attribute]
  headers, grouped = groupFn(cctd, cctd[:,1], ret_col_idx=0)
  val = entropy(sample)
  subtractor = sum(len(grp)/len(sample)*entropy(grp) for grp in grouped)
  return val - subtractor

def split_information(sample, attribute):
  cctd = np.c_[sample, attribute]
  total = len(sample)
  headers, grouped = groupFn(cctd, cctd[:,1], ret_col_idx=0)
  return sum(-float(len(grp)/total)*np.log2(float(len(grp)/total)) for grp in grouped)

def id3(sample, target_attribute, attributes):
  pass

def main():
  sample = sample_data()
  keys, grouped = groupFn(sample, sample[:,1:3], ret_col_idx=0)
  print(entropy(sample[:,2]))
  #gain_ratio_val = gain_ratio(np.array(sample[:, len(sample[0])-1]), np.array(sample[:, 2]))
  #print(tabulate(sample))
  #print(gain_ratio_val)

if __name__ == '__main__':
  main()
