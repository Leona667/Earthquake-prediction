#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Earthquake Group (749-750h) code
# Import the necessary libraries
import h5py
import numpy as np
from scipy.signal import butter, filtfilt


# In[2]:


# Explore the HDF5 file to extract the list of sites and their properties
def explore_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        stations_list = list(f.keys())
        station_attrs = {}
        for station in stations_list:
            attrs = {
                'lat': f[station].attrs.get('lat', None),
                'lon': f[station].attrs.get('lon', None),
                'dist_m': f[station].attrs.get('dist_m', None)
            }
            station_attrs[station] = attrs
        return stations_list, station_attrs


# In[3]:


# Find common stations in multiple files
def find_common_stations(file_paths):
    common_stations = None
    all_station_attrs = {}

    for file_path in file_paths:
        stations_list, station_attrs = explore_h5_file(file_path)
        all_station_attrs[file_path] = station_attrs

        stations_in_file = set(stations_list)

        if common_stations is None:
            common_stations = stations_in_file
        else:
            common_stations = common_stations.intersection(stations_in_file)

    return common_stations, all_station_attrs


# In[4]:


# Extract data for a specified period of time
def extract_data_from_common_stations(common_stations, file_paths, x, y):
    data = {}
    num_channels = 1
    start = x * 3600 * 100
    end = y * 3600 * 100

    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f:
            data[file_path] = {}
            for station in common_stations:
                station_data = f[station][:num_channels, start:end]
                data[file_path][station] = station_data
    
    return data


# In[5]:


# Calculate a list of distances from the epicenter
def distance_epicenter(file_paths):
    distance_list = {}

    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f:
            for station in f.keys():
                distance_list[station] = f[station].attrs['dist_m']
    
    return distance_list


# In[6]:


# Set the file path
file_paths = [
    '/home3/pmzg56/DS Project/ev0000593283.h5',
    '/home3/pmzg56/DS Project/ev0002128689.h5'
]

# Find common stations
common_stations, station_attrs = find_common_stations(file_paths)
print("Common stations across all files:", common_stations)


# In[7]:


# Extract data from public sites, extract 749 to 750th hour data as earthquake group
data = extract_data_from_common_stations(common_stations, file_paths, 749, 750)
print("Extracted data:", data)


# In[8]:


# Calculate the epicenter distance
distances = distance_epicenter(file_paths)
print("Distances from epicenter:", distances)


# In[9]:


# Function for selecting the site
def select_stations_by_distance(distances, num_near=3, num_mid=3, num_far=3):
    # Sort sites by distance
    sorted_stations = sorted(distances.items(), key=lambda item: item[1])
    
    # Select the near range stations
    near_stations = [station for station, _ in sorted_stations[:num_near]]
    
    # Select the medium range stations
    mid_stations = [station for station, _ in sorted_stations[num_near:num_near+num_mid]]
    
    # Select the far range stations
    far_stations = [station for station, _ in sorted_stations[-num_far:]]
    
    return near_stations, mid_stations, far_stations

# Use selection function
near_stations, mid_stations, far_stations = select_stations_by_distance(distances)
print("Near stations:", near_stations)
print("Mid-range stations:", mid_stations)
print("Far stations:", far_stations)

# Select stations and extract corresponding data
selected_stations = near_stations + mid_stations + far_stations

# Extract data and ignore non-existent stations
selected_data = {}

for file_path in data:
    for station in selected_stations:
        if station in data[file_path]:
            if station not in selected_data:
                selected_data[station] = []
            selected_data[station].append(data[file_path][station])

print("Selected data for processing:", selected_data)
print("Available stations in selected data:", list(selected_data.keys()))


# In[10]:


# Define high pass filtering function
def high_pass_filter(data, cutoff_freq=2, fs=100):
    b, a = butter(4, cutoff_freq / (0.5 * fs), btype='high')
    filtered_data = filtfilt(b, a, data, axis=1)
    return filtered_data

# Define signal truncation function
def data_cutoff(data, fs=100, window_size=5):
    num_samples = int(window_size * fs)
    num_windows = data.shape[1] // num_samples
    data_cut = [data[:, i * num_samples:(i + 1) * num_samples] for i in range(num_windows)]
    return np.array(data_cut)


# In[11]:


# Process data for all stations
def process_data(selected_data):
    all_data = []

    for station in selected_data:
        for station_data in selected_data[station]:
            print(f"Processing station {station} with data shape {station_data.shape}")
            cut_data = data_cutoff(station_data)
            print(f"Data shape after slicing: {cut_data.shape}")
            filtered_station_data = np.array([high_pass_filter(chunk) for chunk in cut_data])
            all_data.append(filtered_station_data)
    
    all_data = np.vstack(all_data)
    
    return all_data

filtered_data = process_data(selected_data)
print("Combined data shape:", filtered_data.shape)


# In[12]:


# Fourier Transform
def myfourier(x):
    fs = 100
    n = x.shape[2]
    fi = np.linspace(0, fs / 2, int(n / 2))
    datafft = []

    for i in range(len(x)):
        fft_values = np.abs(np.fft.fft(x[i, 0, :])[:int(n / 2)])
        datafft.append(fft_values)

    return np.array(datafft)

datafft = myfourier(filtered_data)

print("FFT result shape:", datafft.shape)


# In[13]:


import numpy as np
import pandas as pd

# 1.Calculate the characteristics of the integral square waveform
def iosw(x):
    integral = []
    for i in range(len(x)):
        data0 = x[i]
        data_squared = data0 ** 2
        integral.append(np.sum(data_squared))
    return integral

integral = iosw(datafft)

print("Integral of the Squared Waveform (First 10):")
print(integral[:10])


# In[14]:


# 2.Calculate the maximum spectral amplitude and frequency
fs = 100  
n = datafft.shape[1] 
fi = np.linspace(0, fs / 2, n // 2)

def freq(x):
    max_indices = []
    max_amplitudes = []
    for array in x:
        max_index = np.argmax(array)
        max_indices.append(max_index)
        max_amplitudes.append(array[max_index].real)
    
    frequencies = []
    for index in max_indices:
        if index < len(fi):
            frequencies.append(fi[index])
        else:
            frequencies.append(np.nan) 
    
    return frequencies, max_amplitudes

frequencies, max_amplitudes = freq(datafft)

print("Maximum Spectral Amplitude and its Frequency (First 10):")
print("Frequencies:", frequencies[:10])
print("Max Amplitudes:", max_amplitudes[:10])


# In[15]:


# 3.Calculate the center frequency
def cenfreq(x, y):
    center_frequency = []
    for i in range(len(x)):
        fi_resized = np.resize(fi, y[i].shape)
        numerator = np.sum(fi_resized * y[i]).real
        denominator = np.sum(y[i]).real
        center_frequency.append(numerator / denominator)
    return center_frequency

center_frequency = cenfreq(datafft, datafft)

print("Center Frequency (First 10):")
print(center_frequency[:10])


# In[17]:


# 4.Calculate the Signal Bandwidth
def signal_bandwidth(fi, y, z):
    signal_bandwidth = []
    for i in range(len(y)):
        fi_resized = np.resize(fi, y[i].shape)
        numerator = np.sum((fi_resized - z[i])**2)
        denominator = np.sum(z[i])
        bandwidth = np.sqrt(numerator / denominator).real
        signal_bandwidth.append(bandwidth)
    return signal_bandwidth

signal_bandwidth_values = signal_bandwidth(fi, datafft, center_frequency)

print("Signal Bandwidth (First 10):")
print(signal_bandwidth_values[:10])


# In[18]:


# 5.Calculate the Zero Upcrossing Rate
def zero_upcrossing_rate(fi, y):
    zero_upcrossing_rate = []
    for i in range(len(y)):
        fi_resized = np.resize(fi, y[i].shape)
        omega = 2 * np.pi * fi_resized
        numerator = np.sum(omega**2 * y[i]**2)
        denominator = np.sum(y[i])
        z_rate = np.sqrt(numerator / denominator).real
        zero_upcrossing_rate.append(z_rate)
    return zero_upcrossing_rate

zero_upcrossing_rate_values = zero_upcrossing_rate(fi, datafft)

print("Zero Upcrossing Rate (First 10):")
print(zero_upcrossing_rate_values[:10])


# In[19]:


# 6.Calculate the Rate of Spectral Peaks
def rate_of_spectral_peaks(fi, y):
    rate_of_spectral_peaks = []
    for i in range(len(y)):
        fi_resized = np.resize(fi, y[i].shape) 
        omega = 2 * np.pi * fi_resized
        numerator = np.sum(omega**4 * y[i]**2)
        denominator = np.sum(omega**2 * y[i]**2)
        peak_rate = np.sqrt(numerator / denominator).real
        rate_of_spectral_peaks.append(peak_rate)
    return rate_of_spectral_peaks

rate_of_spectral_peaks_values = rate_of_spectral_peaks(fi, datafft)

print("Rate of Spectral Peaks (First 10):")
print(rate_of_spectral_peaks_values[:10])


# In[20]:


# Generate feature table
def creatdf():
    df = pd.DataFrame(list(zip(
        integral, 
        max_amplitudes, 
        frequencies, 
        center_frequency, 
        signal_bandwidth_values, 
        zero_upcrossing_rate_values, 
        rate_of_spectral_peaks_values
    )),
    columns=[
        'integral of the squared waveform', 
        'maximum spectral amplitude', 
        'frequency at the maximum spectral amplitude', 
        'center frequency', 
        'signal bandwidth', 
        'zero upcrossing rate', 
        'rate of spectral peaks'])
    
    return df

df = creatdf()
df


# In[21]:


# Check for missing values
missing_values = df.isnull().sum()

print("Missing values in each column:")
print(missing_values)

if df.isnull().values.any():
    print("The DataFrame contains missing values.")
else:
    print("The DataFrame does not contain any missing values.")


# In[22]:


from sklearn.preprocessing import StandardScaler

# Handling missing values and standardization
df = df.dropna()
scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
missing_values_after_scaling = df_std.isnull().sum()
print("Missing values in each column after standardization:")
print(missing_values_after_scaling)

if df_std.isnull().values.any():
    print("The DataFrame contains missing values after standardization.")
else:
    print("The DataFrame does not contain any missing values after standardization.")
    
print(f"The DataFrame contains {df_std.shape[0]} rows after standardization and removing rows with missing values.")


# In[23]:


# MDS
from sklearn.manifold import MDS

df_std['index'] = df_std.index
df_sample = df_std.sample(frac=0.2572, random_state=42)
mds = MDS(n_components=4, random_state=42)
mds_transformed = mds.fit_transform(df_sample.drop(columns=['index'])) 
df_mds = pd.DataFrame(mds_transformed, columns=[f'Component {i+1}' for i in range(4)])
df_mds['index'] = df_sample['index'].values

print(df_mds.head(10))


# In[24]:


#Elbow Method
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


K = range(2, 21)
Sum_of_squared_distances = []

for num_clusters in K:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(df_mds.drop(columns=['index'])) 
    Sum_of_squared_distances.append(kmeans.inertia_)

plt.figure(figsize=(10, 8))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of squared distances (Inertia)')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.show()


# In[25]:


# K-Means Clustering
from sklearn.cluster import KMeans
import numpy as np

num_clusters = 5
max_iterations = 100
NUM_ATTEMPTS = 500

def bcentroids(data, num_clusters, batch_size, max_iterations, NUM_ATTEMPTS):
    final_cents = []
    final_inert = []

    batch_size = min(batch_size, len(data))

    for sample in range(NUM_ATTEMPTS):
        random_indices = np.random.choice(len(data), size=batch_size, replace=False)
        data_sample = data[random_indices]
        
        km = KMeans(n_clusters=num_clusters, init='random', max_iter=1, n_init=1)
        km.fit(data_sample)
        inertia_start = km.inertia_
        inertia_end = 0
        cents = km.cluster_centers_

        for iter in range(max_iterations):
            km = KMeans(n_clusters=num_clusters, init=cents, max_iter=1, n_init=1)
            km.fit(data_sample)
            inertia_end = km.inertia_
            cents = km.cluster_centers_

        final_cents.append(cents)
        final_inert.append(inertia_end)

    best_cents = final_cents[np.argmin(final_inert)]
    return best_cents

best_cents = bcentroids(df_mds.drop(columns=['index']).values, num_clusters, 15000, max_iterations, NUM_ATTEMPTS)

km_full = KMeans(n_clusters=num_clusters, init=best_cents, max_iter=100, verbose=1, n_init=1)
km_full.fit(df_mds.drop(columns=['index'])) 

labels = km_full.predict(df_mds.drop(columns=['index']))  

cluster_counts = [0] * num_clusters
for label in labels:
    cluster_counts[label] += 1

total_points = len(df_mds)
cluster_percentages = [(count / total_points) * 100 for count in cluster_counts]

for i in range(num_clusters):
    print(f"Label {i+1}: {cluster_percentages[i]:.2f}%")


# In[26]:


from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(df_mds.drop(columns=['index']), labels) 
print(f'Silhouette Score: {silhouette_avg:.4f}')


# In[27]:


# Time series graph of clustering labels
import matplotlib.pyplot as plt
import numpy as np

time_interval_size = 60 
total_intervals = len(df_mds) // time_interval_size

time_series_labels = []

for i in range(total_intervals):
    start_idx = i * time_interval_size
    end_idx = start_idx + time_interval_size
    interval_labels = labels[start_idx:end_idx]
    
    label_counts = [0] * num_clusters
    for label in interval_labels:
        label_counts[label] += 1
    time_series_labels.append([count / time_interval_size * 100 for count in label_counts])

time_series_labels = np.array(time_series_labels)

colors = ['#9467bd', '#2ca02c', '#ff69b4', '#d62728', '#ff7f0e'] 
plt.figure(figsize=(10, 6))

for cluster_id in range(num_clusters):
    plt.plot(range(total_intervals), time_series_labels[:, cluster_id], color=colors[cluster_id], label=f'Label {cluster_id + 1}')

plt.xlabel('Time Intervals')
plt.ylabel('Percentage of Cluster Labels')
plt.title('Cluster Label Distribution Over Time')
plt.legend()
plt.grid(True)
plt.show()

np.save('earthquake_time_series_labels.npy', time_series_labels)


# In[30]:


# Raw waveform _ All Samplings _ SIKH
import matplotlib.pyplot as plt
import numpy as np

station = 'SIKH'
waveform = selected_data[station][0].flatten() * 1e6  
index_mapping = df_sample['index'].values 
label_indices = labels  

fs = 100 

time = np.arange(len(waveform)) / fs  

p_wave_threshold = 0.05  

p_wave_start_index = np.argmax(np.abs(waveform) > p_wave_threshold)
p_wave_start_time = p_wave_start_index / fs

s_wave_start_index = p_wave_start_index + int(5 * fs)  
s_wave_threshold = 0.1 

s_wave_start_index = s_wave_start_index + np.argmax(np.abs(waveform[s_wave_start_index:]) > s_wave_threshold)
s_wave_start_time = s_wave_start_index / fs

print(f"P-wave starts at {p_wave_start_time} seconds")
print(f"S-wave starts at {s_wave_start_time} seconds")

waveform *= 1e6  

plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(waveform)), waveform, color='blue', alpha=0.5)  

plt.xlabel('Sampling Point')  
plt.ylabel('Amplitude ($10^6$ units)') 
plt.title(f'Raw Waveform Snapshot for Station {station}')
plt.grid(True)
plt.show()


# In[31]:


# Raw waveform _ All Labels Samplings _ SIKH
import matplotlib.pyplot as plt
import numpy as np

station = 'SIKH'
waveform = selected_data[station][0].flatten() * 1e6  
index_mapping = df_sample['index'].values
label_indices = labels 

fs = 100  

p_wave_threshold = 0.05 
p_wave_start_index = np.argmax(np.abs(waveform) > p_wave_threshold)
p_wave_start_time = p_wave_start_index / fs

s_wave_start_index = p_wave_start_index + int(5 * fs)  
s_wave_threshold = 0.1  
s_wave_start_index = s_wave_start_index + np.argmax(np.abs(waveform[s_wave_start_index:]) > s_wave_threshold)
s_wave_start_time = s_wave_start_index / fs

print(f"P-wave starts at {p_wave_start_time} seconds")
print(f"S-wave starts at {s_wave_start_time} seconds")

plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(waveform)), waveform, color='blue', alpha=0.5) 

colors = ['#9467bd', '#2ca02c', '#ff69b4', '#d62728', '#ff7f0e'] 

for label in np.unique(label_indices):
    label_mask = (label_indices == label)
    mapped_indices = index_mapping[label_mask]
    mapped_indices = mapped_indices[mapped_indices < len(waveform)] 
    plt.plot(mapped_indices, waveform[mapped_indices], color=colors[label], alpha=1.0, label=f'Label {label + 1}')

plt.xlabel('Sampling Point')  
plt.ylabel('Amplitude ($10^6$ units)')  
plt.title(f'Clustered Waveform Snapshot for Station {station}')
plt.legend()
plt.grid(True)
plt.show()


# In[32]:


# Labels waveform _ P waves and S waves _ SIKH
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d 

station = 'SIKH' 
waveform = selected_data[station][0].flatten() * 1e6 
index_mapping = df_sample['index'].values 
label_indices = labels 

fs = 100
p_wave_threshold = 0.05 
p_wave_start_index = np.argmax(np.abs(waveform) > p_wave_threshold)
p_wave_start_time = p_wave_start_index / fs

s_wave_start_index = p_wave_start_index + int(5 * fs) 
s_wave_threshold = 0.1 
s_wave_start_index = s_wave_start_index + np.argmax(np.abs(waveform[s_wave_start_index:]) > s_wave_threshold)
s_wave_start_time = s_wave_start_index / fs

print(f"P-wave starts at {p_wave_start_time} seconds")
print(f"S-wave starts at {s_wave_start_time} seconds")

p_wave_window_duration = 3  
s_wave_window_duration = 3 

p_wave_start_index = max(0, p_wave_start_index)
p_wave_end_index = min(len(waveform), p_wave_start_index + int(p_wave_window_duration * fs))

s_wave_start_index = max(0, s_wave_start_index)
s_wave_end_index = min(len(waveform), s_wave_start_index + int(s_wave_window_duration * fs))

waveform_smooth = gaussian_filter1d(waveform, sigma=2) 

p_waveform_window = waveform_smooth[p_wave_start_index:p_wave_end_index]
s_waveform_window = waveform_smooth[s_wave_start_index:s_wave_end_index]
p_time_window = np.arange(p_wave_start_index, p_wave_end_index)
s_time_window = np.arange(s_wave_start_index, s_wave_end_index)

def calculate_label_proportions(start_index, end_index, label_indices, index_mapping):
    total_points_in_window = end_index - start_index
    label_proportions = {}

    for label in np.unique(label_indices):
        label_mask = (label_indices == label)
        mapped_indices = index_mapping[label_mask]
        mapped_indices = mapped_indices[(mapped_indices >= start_index) & (mapped_indices < end_index)]
        mapped_indices = mapped_indices[mapped_indices < len(waveform)] 
        label_count = len(mapped_indices)
        label_proportions[label] = (label_count / total_points_in_window) * 100

    return label_proportions

p_wave_label_proportions = calculate_label_proportions(p_wave_start_index, p_wave_end_index, label_indices, index_mapping)
s_wave_label_proportions = calculate_label_proportions(s_wave_start_index, s_wave_end_index, label_indices, index_mapping)

print("P-wave label proportions:")
for label, proportion in p_wave_label_proportions.items():
    print(f"Label {label + 1}: {proportion:.2f}%")

print("\nS-wave label proportions:")
for label, proportion in s_wave_label_proportions.items():
    print(f"Label {label + 1}: {proportion:.2f}%")

colors = ['#9467bd', '#2ca02c', '#ff69b4', '#d62728', '#ff7f0e'] 

plt.figure(figsize=(10, 6))

for label in np.unique(label_indices):
    label_mask = (label_indices == label)
    mapped_indices = index_mapping[label_mask]
   
    mapped_indices = mapped_indices[(mapped_indices >= p_wave_start_index) & (mapped_indices < p_wave_end_index)]
    mapped_indices = mapped_indices[mapped_indices < len(waveform)]  
    plt.plot(mapped_indices, waveform[mapped_indices], color=colors[label], alpha=0.6, label=f'P-wave Label {label + 1}')


for label in np.unique(label_indices):
    label_mask = (label_indices == label)
    mapped_indices = index_mapping[label_mask]
    
    mapped_indices = mapped_indices[(mapped_indices >= s_wave_start_index) & (mapped_indices < s_wave_end_index)]
    mapped_indices = mapped_indices[mapped_indices < len(waveform)]  
    plt.plot(mapped_indices, waveform[mapped_indices], color=colors[label], alpha=0.6, linestyle='--', label=f'S-wave Label {label + 1}')


plt.axvline(x=p_wave_start_index, color='green', linestyle='--', label=f'P-wave Start ({p_wave_start_time:.2f}s)')
plt.axvline(x=s_wave_start_index, color='red', linestyle='--', label=f'S-wave Start ({s_wave_start_time:.2f}s)')
plt.xlabel('Sampling Point') 
plt.ylabel('Amplitude ($10^6$ units)') 
plt.title(f'Clustered Waveform Snapshot for Station {station} (P-wave and S-wave)')
plt.legend()
plt.grid(True)
plt.show()


# In[33]:


# Raw waveform _ All Samplings _ KIGH
import matplotlib.pyplot as plt
import numpy as np

station = 'KIGH'
waveform = selected_data[station][0].flatten() * 1e6  
index_mapping = df_sample['index'].values 
label_indices = labels  

fs = 100 

time = np.arange(len(waveform)) / fs  

p_wave_threshold = 0.05  

p_wave_start_index = np.argmax(np.abs(waveform) > p_wave_threshold)
p_wave_start_time = p_wave_start_index / fs

s_wave_start_index = p_wave_start_index + int(5 * fs)  
s_wave_threshold = 0.1 

s_wave_start_index = s_wave_start_index + np.argmax(np.abs(waveform[s_wave_start_index:]) > s_wave_threshold)
s_wave_start_time = s_wave_start_index / fs

print(f"P-wave starts at {p_wave_start_time} seconds")
print(f"S-wave starts at {s_wave_start_time} seconds")

waveform *= 1e6  

plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(waveform)), waveform, color='blue', alpha=0.5)  

plt.xlabel('Sampling Point')  
plt.ylabel('Amplitude ($10^6$ units)') 
plt.title(f'Raw Waveform Snapshot for Station {station}')
plt.grid(True)
plt.show()


# In[34]:


# Raw waveform _ All Labels Samplings _ KIGH
import matplotlib.pyplot as plt
import numpy as np

station = 'KIGH'
waveform = selected_data[station][0].flatten() * 1e6  
index_mapping = df_sample['index'].values
label_indices = labels 

fs = 100  

p_wave_threshold = 0.05 
p_wave_start_index = np.argmax(np.abs(waveform) > p_wave_threshold)
p_wave_start_time = p_wave_start_index / fs

s_wave_start_index = p_wave_start_index + int(5 * fs)  
s_wave_threshold = 0.1  
s_wave_start_index = s_wave_start_index + np.argmax(np.abs(waveform[s_wave_start_index:]) > s_wave_threshold)
s_wave_start_time = s_wave_start_index / fs

print(f"P-wave starts at {p_wave_start_time} seconds")
print(f"S-wave starts at {s_wave_start_time} seconds")

plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(waveform)), waveform, color='blue', alpha=0.5) 

colors = ['#9467bd', '#2ca02c', '#ff69b4', '#d62728', '#ff7f0e'] 

for label in np.unique(label_indices):
    label_mask = (label_indices == label)
    mapped_indices = index_mapping[label_mask]
    mapped_indices = mapped_indices[mapped_indices < len(waveform)] 
    plt.plot(mapped_indices, waveform[mapped_indices], color=colors[label], alpha=1.0, label=f'Label {label + 1}')

plt.xlabel('Sampling Point')  
plt.ylabel('Amplitude ($10^6$ units)')  
plt.title(f'Clustered Waveform Snapshot for Station {station}')
plt.legend()
plt.grid(True)
plt.show()


# In[35]:


# Labels waveform _ P waves and S waves _ KIGH
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d 

station = 'KIGH' 
waveform = selected_data[station][0].flatten() * 1e6 
index_mapping = df_sample['index'].values 
label_indices = labels 

fs = 100
p_wave_threshold = 0.05 
p_wave_start_index = np.argmax(np.abs(waveform) > p_wave_threshold)
p_wave_start_time = p_wave_start_index / fs

s_wave_start_index = p_wave_start_index + int(5 * fs) 
s_wave_threshold = 0.1 
s_wave_start_index = s_wave_start_index + np.argmax(np.abs(waveform[s_wave_start_index:]) > s_wave_threshold)
s_wave_start_time = s_wave_start_index / fs

print(f"P-wave starts at {p_wave_start_time} seconds")
print(f"S-wave starts at {s_wave_start_time} seconds")

p_wave_window_duration = 3  
s_wave_window_duration = 3 

p_wave_start_index = max(0, p_wave_start_index)
p_wave_end_index = min(len(waveform), p_wave_start_index + int(p_wave_window_duration * fs))

s_wave_start_index = max(0, s_wave_start_index)
s_wave_end_index = min(len(waveform), s_wave_start_index + int(s_wave_window_duration * fs))

waveform_smooth = gaussian_filter1d(waveform, sigma=2) 

p_waveform_window = waveform_smooth[p_wave_start_index:p_wave_end_index]
s_waveform_window = waveform_smooth[s_wave_start_index:s_wave_end_index]
p_time_window = np.arange(p_wave_start_index, p_wave_end_index)
s_time_window = np.arange(s_wave_start_index, s_wave_end_index)

def calculate_label_proportions(start_index, end_index, label_indices, index_mapping):
    total_points_in_window = end_index - start_index
    label_proportions = {}

    for label in np.unique(label_indices):
        label_mask = (label_indices == label)
        mapped_indices = index_mapping[label_mask]
        mapped_indices = mapped_indices[(mapped_indices >= start_index) & (mapped_indices < end_index)]
        mapped_indices = mapped_indices[mapped_indices < len(waveform)] 
        label_count = len(mapped_indices)
        label_proportions[label] = (label_count / total_points_in_window) * 100

    return label_proportions

p_wave_label_proportions = calculate_label_proportions(p_wave_start_index, p_wave_end_index, label_indices, index_mapping)
s_wave_label_proportions = calculate_label_proportions(s_wave_start_index, s_wave_end_index, label_indices, index_mapping)

print("P-wave label proportions:")
for label, proportion in p_wave_label_proportions.items():
    print(f"Label {label + 1}: {proportion:.2f}%")

print("\nS-wave label proportions:")
for label, proportion in s_wave_label_proportions.items():
    print(f"Label {label + 1}: {proportion:.2f}%")

colors = ['#9467bd', '#2ca02c', '#ff69b4', '#d62728', '#ff7f0e'] 

plt.figure(figsize=(10, 6))

for label in np.unique(label_indices):
    label_mask = (label_indices == label)
    mapped_indices = index_mapping[label_mask]
   
    mapped_indices = mapped_indices[(mapped_indices >= p_wave_start_index) & (mapped_indices < p_wave_end_index)]
    mapped_indices = mapped_indices[mapped_indices < len(waveform)]  
    plt.plot(mapped_indices, waveform[mapped_indices], color=colors[label], alpha=0.6, label=f'P-wave Label {label + 1}')


for label in np.unique(label_indices):
    label_mask = (label_indices == label)
    mapped_indices = index_mapping[label_mask]
    
    mapped_indices = mapped_indices[(mapped_indices >= s_wave_start_index) & (mapped_indices < s_wave_end_index)]
    mapped_indices = mapped_indices[mapped_indices < len(waveform)]  
    plt.plot(mapped_indices, waveform[mapped_indices], color=colors[label], alpha=0.6, linestyle='--', label=f'S-wave Label {label + 1}')


plt.axvline(x=p_wave_start_index, color='green', linestyle='--', label=f'P-wave Start ({p_wave_start_time:.2f}s)')
plt.axvline(x=s_wave_start_index, color='red', linestyle='--', label=f'S-wave Start ({s_wave_start_time:.2f}s)')
plt.xlabel('Sampling Point') 
plt.ylabel('Amplitude ($10^6$ units)') 
plt.title(f'Clustered Waveform Snapshot for Station {station} (P-wave and S-wave)')
plt.legend()
plt.grid(True)
plt.show()


# In[36]:


# Labels waveform _ P waves and S waves _ UMWH
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d 

station = 'UMWH' 
waveform = selected_data[station][0].flatten() * 1e6 
index_mapping = df_sample['index'].values 
label_indices = labels 

fs = 100
p_wave_threshold = 0.05 
p_wave_start_index = np.argmax(np.abs(waveform) > p_wave_threshold)
p_wave_start_time = p_wave_start_index / fs

s_wave_start_index = p_wave_start_index + int(5 * fs) 
s_wave_threshold = 0.1 
s_wave_start_index = s_wave_start_index + np.argmax(np.abs(waveform[s_wave_start_index:]) > s_wave_threshold)
s_wave_start_time = s_wave_start_index / fs

print(f"P-wave starts at {p_wave_start_time} seconds")
print(f"S-wave starts at {s_wave_start_time} seconds")

p_wave_window_duration = 3  
s_wave_window_duration = 3 

p_wave_start_index = max(0, p_wave_start_index)
p_wave_end_index = min(len(waveform), p_wave_start_index + int(p_wave_window_duration * fs))

s_wave_start_index = max(0, s_wave_start_index)
s_wave_end_index = min(len(waveform), s_wave_start_index + int(s_wave_window_duration * fs))

waveform_smooth = gaussian_filter1d(waveform, sigma=2) 

p_waveform_window = waveform_smooth[p_wave_start_index:p_wave_end_index]
s_waveform_window = waveform_smooth[s_wave_start_index:s_wave_end_index]
p_time_window = np.arange(p_wave_start_index, p_wave_end_index)
s_time_window = np.arange(s_wave_start_index, s_wave_end_index)

def calculate_label_proportions(start_index, end_index, label_indices, index_mapping):
    total_points_in_window = end_index - start_index
    label_proportions = {}

    for label in np.unique(label_indices):
        label_mask = (label_indices == label)
        mapped_indices = index_mapping[label_mask]
        mapped_indices = mapped_indices[(mapped_indices >= start_index) & (mapped_indices < end_index)]
        mapped_indices = mapped_indices[mapped_indices < len(waveform)] 
        label_count = len(mapped_indices)
        label_proportions[label] = (label_count / total_points_in_window) * 100

    return label_proportions

p_wave_label_proportions = calculate_label_proportions(p_wave_start_index, p_wave_end_index, label_indices, index_mapping)
s_wave_label_proportions = calculate_label_proportions(s_wave_start_index, s_wave_end_index, label_indices, index_mapping)

print("P-wave label proportions:")
for label, proportion in p_wave_label_proportions.items():
    print(f"Label {label + 1}: {proportion:.2f}%")

print("\nS-wave label proportions:")
for label, proportion in s_wave_label_proportions.items():
    print(f"Label {label + 1}: {proportion:.2f}%")

colors = ['#9467bd', '#2ca02c', '#ff69b4', '#d62728', '#ff7f0e'] 

plt.figure(figsize=(10, 6))

for label in np.unique(label_indices):
    label_mask = (label_indices == label)
    mapped_indices = index_mapping[label_mask]
   
    mapped_indices = mapped_indices[(mapped_indices >= p_wave_start_index) & (mapped_indices < p_wave_end_index)]
    mapped_indices = mapped_indices[mapped_indices < len(waveform)]  
    plt.plot(mapped_indices, waveform[mapped_indices], color=colors[label], alpha=0.6, label=f'P-wave Label {label + 1}')


for label in np.unique(label_indices):
    label_mask = (label_indices == label)
    mapped_indices = index_mapping[label_mask]
    
    mapped_indices = mapped_indices[(mapped_indices >= s_wave_start_index) & (mapped_indices < s_wave_end_index)]
    mapped_indices = mapped_indices[mapped_indices < len(waveform)]  
    plt.plot(mapped_indices, waveform[mapped_indices], color=colors[label], alpha=0.6, linestyle='--', label=f'S-wave Label {label + 1}')


plt.axvline(x=p_wave_start_index, color='green', linestyle='--', label=f'P-wave Start ({p_wave_start_time:.2f}s)')
plt.axvline(x=s_wave_start_index, color='red', linestyle='--', label=f'S-wave Start ({s_wave_start_time:.2f}s)')
plt.xlabel('Sampling Point') 
plt.ylabel('Amplitude ($10^6$ units)') 
plt.title(f'Clustered Waveform Snapshot for Station {station} (P-wave and S-wave)')
plt.legend()
plt.grid(True)
plt.show()


# In[37]:


# Labels waveform _ P waves and S waves _ ASKH
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d 

station = 'ASKH' 
waveform = selected_data[station][0].flatten() * 1e6 
index_mapping = df_sample['index'].values 
label_indices = labels 

fs = 100
p_wave_threshold = 0.05 
p_wave_start_index = np.argmax(np.abs(waveform) > p_wave_threshold)
p_wave_start_time = p_wave_start_index / fs

s_wave_start_index = p_wave_start_index + int(5 * fs) 
s_wave_threshold = 0.1 
s_wave_start_index = s_wave_start_index + np.argmax(np.abs(waveform[s_wave_start_index:]) > s_wave_threshold)
s_wave_start_time = s_wave_start_index / fs

print(f"P-wave starts at {p_wave_start_time} seconds")
print(f"S-wave starts at {s_wave_start_time} seconds")

p_wave_window_duration = 3  
s_wave_window_duration = 3 

p_wave_start_index = max(0, p_wave_start_index)
p_wave_end_index = min(len(waveform), p_wave_start_index + int(p_wave_window_duration * fs))

s_wave_start_index = max(0, s_wave_start_index)
s_wave_end_index = min(len(waveform), s_wave_start_index + int(s_wave_window_duration * fs))

waveform_smooth = gaussian_filter1d(waveform, sigma=2) 

p_waveform_window = waveform_smooth[p_wave_start_index:p_wave_end_index]
s_waveform_window = waveform_smooth[s_wave_start_index:s_wave_end_index]
p_time_window = np.arange(p_wave_start_index, p_wave_end_index)
s_time_window = np.arange(s_wave_start_index, s_wave_end_index)

def calculate_label_proportions(start_index, end_index, label_indices, index_mapping):
    total_points_in_window = end_index - start_index
    label_proportions = {}

    for label in np.unique(label_indices):
        label_mask = (label_indices == label)
        mapped_indices = index_mapping[label_mask]
        mapped_indices = mapped_indices[(mapped_indices >= start_index) & (mapped_indices < end_index)]
        mapped_indices = mapped_indices[mapped_indices < len(waveform)] 
        label_count = len(mapped_indices)
        label_proportions[label] = (label_count / total_points_in_window) * 100

    return label_proportions

p_wave_label_proportions = calculate_label_proportions(p_wave_start_index, p_wave_end_index, label_indices, index_mapping)
s_wave_label_proportions = calculate_label_proportions(s_wave_start_index, s_wave_end_index, label_indices, index_mapping)

print("P-wave label proportions:")
for label, proportion in p_wave_label_proportions.items():
    print(f"Label {label + 1}: {proportion:.2f}%")

print("\nS-wave label proportions:")
for label, proportion in s_wave_label_proportions.items():
    print(f"Label {label + 1}: {proportion:.2f}%")

colors = ['#9467bd', '#2ca02c', '#ff69b4', '#d62728', '#ff7f0e'] 

plt.figure(figsize=(10, 6))

for label in np.unique(label_indices):
    label_mask = (label_indices == label)
    mapped_indices = index_mapping[label_mask]
   
    mapped_indices = mapped_indices[(mapped_indices >= p_wave_start_index) & (mapped_indices < p_wave_end_index)]
    mapped_indices = mapped_indices[mapped_indices < len(waveform)]  
    plt.plot(mapped_indices, waveform[mapped_indices], color=colors[label], alpha=0.6, label=f'P-wave Label {label + 1}')


for label in np.unique(label_indices):
    label_mask = (label_indices == label)
    mapped_indices = index_mapping[label_mask]
    
    mapped_indices = mapped_indices[(mapped_indices >= s_wave_start_index) & (mapped_indices < s_wave_end_index)]
    mapped_indices = mapped_indices[mapped_indices < len(waveform)]  
    plt.plot(mapped_indices, waveform[mapped_indices], color=colors[label], alpha=0.6, linestyle='--', label=f'S-wave Label {label + 1}')


plt.axvline(x=p_wave_start_index, color='green', linestyle='--', label=f'P-wave Start ({p_wave_start_time:.2f}s)')
plt.axvline(x=s_wave_start_index, color='red', linestyle='--', label=f'S-wave Start ({s_wave_start_time:.2f}s)')
plt.xlabel('Sampling Point') 
plt.ylabel('Amplitude ($10^6$ units)') 
plt.title(f'Clustered Waveform Snapshot for Station {station} (P-wave and S-wave)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# Save centroid
earthquake_centroids = km_full.cluster_centers_
print("Earthquake Centroids (Shape: {}):".format(earthquake_centroids.shape))
print(earthquake_centroids)
np.save('earthquake_centroids.npy', earthquake_centroids)


# In[ ]:


## Earthquake Group (749-750h) code
# Extract data from public sites and extract data from the 650th to 662th hour as non-earthquake groups
data = extract_data_from_common_stations(common_stations, file_paths, 650, 662)
print("Extracted data:", data)


# In[ ]:


# Calculate the epicenter distance
distances = distance_epicenter(file_paths)
print("Distances from epicenter:", distances)


# In[ ]:


# Function for selecting the site
def select_stations_by_distance(distances, num_near=3, num_mid=3, num_far=3):
    # Sort sites by distance
    sorted_stations = sorted(distances.items(), key=lambda item: item[1])
    
    # Select the near range stations
    near_stations = [station for station, _ in sorted_stations[:num_near]]
    
    # Select the medium range stations
    mid_stations = [station for station, _ in sorted_stations[num_near:num_near+num_mid]]
    
    # Select the far range stations
    far_stations = [station for station, _ in sorted_stations[-num_far:]]
    
    return near_stations, mid_stations, far_stations

# Use selection function
near_stations, mid_stations, far_stations = select_stations_by_distance(distances)
print("Near stations:", near_stations)
print("Mid-range stations:", mid_stations)
print("Far stations:", far_stations)

# Select stations and extract corresponding data
selected_stations = near_stations + mid_stations + far_stations

# Extract data and ignore non-existent stations
selected_data = {}

for file_path in data:
    for station in selected_stations:
        if station in data[file_path]:
            if station not in selected_data:
                selected_data[station] = []
            selected_data[station].append(data[file_path][station])

print("Selected data for processing:", selected_data)
print("Available stations in selected data:", list(selected_data.keys()))


# In[ ]:


# Define high pass filtering function
def high_pass_filter(data, cutoff_freq=2, fs=100):
    b, a = butter(4, cutoff_freq / (0.5 * fs), btype='high')
    filtered_data = filtfilt(b, a, data, axis=1)
    return filtered_data

# Define signal truncation function
def data_cutoff(data, fs=100, window_size=5):
    num_samples = int(window_size * fs)
    num_windows = data.shape[1] // num_samples
    data_cut = [data[:, i * num_samples:(i + 1) * num_samples] for i in range(num_windows)]
    return np.array(data_cut)

# Process data for all stations
def process_data(selected_data):
    all_data = []

    for station in selected_data:
        for station_data in selected_data[station]:
            print(f"Processing station {station} with data shape {station_data.shape}")
            cut_data = data_cutoff(station_data)
            print(f"Data shape after slicing: {cut_data.shape}")
            filtered_station_data = np.array([high_pass_filter(chunk) for chunk in cut_data])
            all_data.append(filtered_station_data)
    
    all_data = np.vstack(all_data)
    
    return all_data

filtered_data = process_data(selected_data)
print("Combined data shape:", filtered_data.shape)


# In[ ]:


# Fourier Transform
def myfourier(x):
    fs = 100
    n = x.shape[2]
    fi = np.linspace(0, fs / 2, int(n / 2))
    datafft = []

    for i in range(len(x)):
        fft_values = np.abs(np.fft.fft(x[i, 0, :])[:int(n / 2)])
        datafft.append(fft_values)

    return np.array(datafft)

datafft = myfourier(filtered_data)

print("FFT result shape:", datafft.shape)


# In[ ]:


# 1.Calculate the characteristics of the integral square waveform
def iosw(x):
    integral = []
    for i in range(len(x)):
        data0 = x[i]
        data_squared = data0 ** 2
        integral.append(np.sum(data_squared))
    return integral

integral = iosw(datafft)

# 2.Calculate the maximum spectral amplitude and frequency
fs = 100  
n = datafft.shape[1] 
fi = np.linspace(0, fs / 2, n // 2)

def freq(x):
    max_indices = []
    max_amplitudes = []
    for array in x:
        max_index = np.argmax(array)
        max_indices.append(max_index)
        max_amplitudes.append(array[max_index].real)
    
    frequencies = []
    for index in max_indices:
        if index < len(fi):
            frequencies.append(fi[index])
        else:
            frequencies.append(np.nan) 
    
    return frequencies, max_amplitudes

frequencies, max_amplitudes = freq(datafft)

# 3.Calculate the center frequency
def cenfreq(x, y):
    center_frequency = []
    for i in range(len(x)):
        fi_resized = np.resize(fi, y[i].shape)
        numerator = np.sum(fi_resized * y[i]).real
        denominator = np.sum(y[i]).real
        center_frequency.append(numerator / denominator)
    return center_frequency

center_frequency = cenfreq(datafft, datafft)

# 4.Calculate the Signal Bandwidth
def signal_bandwidth(fi, y, z):
    signal_bandwidth = []
    for i in range(len(y)):
        fi_resized = np.resize(fi, y[i].shape)
        numerator = np.sum((fi_resized - z[i])**2)
        denominator = np.sum(z[i])
        bandwidth = np.sqrt(numerator / denominator).real
        signal_bandwidth.append(bandwidth)
    return signal_bandwidth

signal_bandwidth_values = signal_bandwidth(fi, datafft, center_frequency)

# 5.Calculate the Zero Upcrossing Rate
def zero_upcrossing_rate(fi, y):
    zero_upcrossing_rate = []
    for i in range(len(y)):
        fi_resized = np.resize(fi, y[i].shape)
        omega = 2 * np.pi * fi_resized
        numerator = np.sum(omega**2 * y[i]**2)
        denominator = np.sum(y[i])
        z_rate = np.sqrt(numerator / denominator).real
        zero_upcrossing_rate.append(z_rate)
    return zero_upcrossing_rate

zero_upcrossing_rate_values = zero_upcrossing_rate(fi, datafft)

# 6.Calculate the Rate of Spectral Peaks
def rate_of_spectral_peaks(fi, y):
    rate_of_spectral_peaks = []
    for i in range(len(y)):
        fi_resized = np.resize(fi, y[i].shape) 
        omega = 2 * np.pi * fi_resized
        numerator = np.sum(omega**4 * y[i]**2)
        denominator = np.sum(omega**2 * y[i]**2)
        peak_rate = np.sqrt(numerator / denominator).real
        rate_of_spectral_peaks.append(peak_rate)
    return rate_of_spectral_peaks

rate_of_spectral_peaks_values = rate_of_spectral_peaks(fi, datafft)

# Generate feature table
def creatdf():
    df = pd.DataFrame(list(zip(
        integral, 
        max_amplitudes, 
        frequencies, 
        center_frequency, 
        signal_bandwidth_values, 
        zero_upcrossing_rate_values, 
        rate_of_spectral_peaks_values
    )),
    columns=[
        'integral of the squared waveform', 
        'maximum spectral amplitude', 
        'frequency at the maximum spectral amplitude', 
        'center frequency', 
        'signal bandwidth', 
        'zero upcrossing rate', 
        'rate of spectral peaks'])
    
    return df

df = creatdf()
df


# In[ ]:


# Check for missing values
missing_values = df.isnull().sum()

print("Missing values in each column:")
print(missing_values)

if df.isnull().values.any():
    print("The DataFrame contains missing values.")
else:
    print("The DataFrame does not contain any missing values.")

from sklearn.preprocessing import StandardScaler

# Handling missing values and standardization
df = df.dropna()
scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
missing_values_after_scaling = df_std.isnull().sum()
print("Missing values in each column after standardization:")
print(missing_values_after_scaling)

if df_std.isnull().values.any():
    print("The DataFrame contains missing values after standardization.")
else:
    print("The DataFrame does not contain any missing values after standardization.")
    
print(f"The DataFrame contains {df_std.shape[0]} rows after standardization and removing rows with missing values.")


# In[ ]:


# MDS
from sklearn.manifold import MDS

df_std['index'] = df_std.index

df_sample = df_std.sample(frac=0.02, random_state=42)

mds = MDS(n_components=4, random_state=42)
mds_transformed = mds.fit_transform(df_sample.drop(columns=['index']))

df_mds = pd.DataFrame(mds_transformed, columns=[f'Component {i+1}' for i in range(4)])
df_mds['index'] = df_sample['index'].values 

print(df_mds.head(10))


# In[ ]:


# Elbow Method 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

K = range(2, 21)

Sum_of_squared_distances = []

for num_clusters in K:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(df_mds.drop(columns=['index']))
    Sum_of_squared_distances.append(kmeans.inertia_)

plt.figure(figsize=(10, 8))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of squared distances (Inertia)')
plt.title('Elbow Method For Optimal k _ Non Earthquake Group')
plt.grid(True)
plt.show()


# In[ ]:


# K-Means Clustering
from sklearn.cluster import KMeans
import numpy as np

num_clusters = 5
max_iterations = 100
NUM_ATTEMPTS = 500

def bcentroids(data, num_clusters, batch_size, max_iterations, NUM_ATTEMPTS):
    final_cents = []
    final_inert = []

    batch_size = min(batch_size, len(data))

    for sample in range(NUM_ATTEMPTS):
        random_indices = np.random.choice(len(data), size=batch_size, replace=False)
        data_sample = data[random_indices]
        
        km = KMeans(n_clusters=num_clusters, init='random', max_iter=1, n_init=1)
        km.fit(data_sample)
        inertia_start = km.inertia_
        inertia_end = 0
        cents = km.cluster_centers_

        for iter in range(max_iterations):
            km = KMeans(n_clusters=num_clusters, init=cents, max_iter=1, n_init=1)
            km.fit(data_sample)
            inertia_end = km.inertia_
            cents = km.cluster_centers_

        final_cents.append(cents)
        final_inert.append(inertia_end)

    best_cents = final_cents[np.argmin(final_inert)]
    return best_cents

best_cents = bcentroids(df_mds.drop(columns=['index']).values, num_clusters, 15000, max_iterations, NUM_ATTEMPTS)

km_full = KMeans(n_clusters=num_clusters, init=best_cents, max_iter=100, verbose=1, n_init=1)
km_full.fit(df_mds.drop(columns=['index'])) 

labels = km_full.predict(df_mds.drop(columns=['index']))  # 去除 'index' 列

cluster_counts = [0] * num_clusters
for label in labels:
    cluster_counts[label] += 1

total_points = len(df_mds)
cluster_percentages = [(count / total_points) * 100 for count in cluster_counts]

for i in range(num_clusters):
    print(f"Label {i+1}: {cluster_percentages[i]:.2f}%")


# In[ ]:


# Time series graph of clustering labels
import matplotlib.pyplot as plt
import numpy as np

time_interval_size = 60  
total_intervals = len(df_mds) // time_interval_size

time_series_labels = []

for i in range(total_intervals):
    start_idx = i * time_interval_size
    end_idx = start_idx + time_interval_size
    interval_labels = labels[start_idx:end_idx]
    
    label_counts = [0] * num_clusters
    for label in interval_labels:
        label_counts[label] += 1
    time_series_labels.append([count / time_interval_size * 100 for count in label_counts]) 

time_series_labels = np.array(time_series_labels)

colors = ['#9467bd', '#2ca02c', '#ff69b4', '#d62728', '#ff7f0e'] 
plt.figure(figsize=(10, 6))

for cluster_id in range(num_clusters):
    plt.plot(range(total_intervals), time_series_labels[:, cluster_id], color=colors[cluster_id], label=f'Label {cluster_id + 1}')

plt.xlabel('Time Intervals')
plt.ylabel('Percentage of Cluster Labels')
plt.title('Cluster Label Distribution Over Time _ Non Earthquake Group')
plt.legend()
plt.grid(True)
plt.show()

np.save('non_earthquake_time_series_labels.npy', time_series_labels)


# In[ ]:


# Raw waveform _ All Samplings _ KIGH
import matplotlib.pyplot as plt
import numpy as np

station = 'SIKH'  
waveform = selected_data[station][0].flatten() * 1e6  
index_mapping = df_sample['index'].values  
label_indices = labels 

fs = 100  

sampling_rate = 10
waveform_downsampled = waveform[::sampling_rate]
downsampled_indices = np.arange(0, len(waveform), sampling_rate)

p_wave_threshold = 0.05
p_wave_start_index = np.argmax(np.abs(waveform_downsampled) > p_wave_threshold)
p_wave_start_time = (p_wave_start_index * sampling_rate) / fs  

s_wave_start_index = p_wave_start_index + int(5 * fs / sampling_rate) 
s_wave_threshold = 0.1  
s_wave_start_index = s_wave_start_index + np.argmax(np.abs(waveform_downsampled[s_wave_start_index:]) > s_wave_threshold)
s_wave_start_time = (s_wave_start_index * sampling_rate) / fs 

print(f"P-wave starts at {p_wave_start_time:.2f} seconds (sampling point: {p_wave_start_index * sampling_rate})")
print(f"S-wave starts at {s_wave_start_time:.2f} seconds (sampling point: {s_wave_start_index * sampling_rate})")

plt.figure(figsize=(10, 6))
plt.plot(downsampled_indices, waveform_downsampled, color='blue', alpha=0.5)
plt.axvline(x=p_wave_start_index * sampling_rate, color='green', linestyle='--', label=f'P-wave Start ({p_wave_start_time:.2f}s)')
plt.axvline(x=s_wave_start_index * sampling_rate, color='red', linestyle='--', label=f'S-wave Start ({s_wave_start_time:.2f}s)')

plt.xlabel('Sampling Point') 
plt.ylabel('Amplitude ($10^6$ units)')
plt.title(f'Raw Waveform Snapshot for Station {station}')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# Labels waveform _ P waves and S waves _ SIKH
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

station = 'SIKH'  
waveform = selected_data[station][0].flatten() * 1e6  
index_mapping = df_sample['index'].values  
label_indices = labels  

fs = 100 


p_wave_threshold = 0.05  
p_wave_start_index = np.argmax(np.abs(waveform) > p_wave_threshold)
p_wave_start_time = p_wave_start_index / fs

s_wave_start_index = p_wave_start_index + int(5 * fs) 
s_wave_threshold = 0.1  
s_wave_start_index = s_wave_start_index + np.argmax(np.abs(waveform[s_wave_start_index:]) > s_wave_threshold)
s_wave_start_time = s_wave_start_index / fs

print(f"P-wave starts at {p_wave_start_time} seconds")
print(f"S-wave starts at {s_wave_start_time} seconds")

window_duration = 3 
p_wave_start_index = max(0, p_wave_start_index)
p_wave_end_index = min(len(waveform), p_wave_start_index + int(window_duration * fs))

s_wave_start_index = max(0, s_wave_start_index)
s_wave_end_index = min(len(waveform), s_wave_start_index + int(window_duration * fs))

waveform_smooth = gaussian_filter1d(waveform, sigma=2) 

p_time_window = np.arange(p_wave_start_index, p_wave_end_index)
s_time_window = np.arange(s_wave_start_index, s_wave_end_index)

def calculate_label_proportions(time_window, start_index, end_index, label_indices, index_mapping):
    total_points_in_window = len(time_window)
    label_proportions = {}

    for label in np.unique(label_indices):
        label_mask = (label_indices == label)
        mapped_indices = index_mapping[label_mask]
        mapped_indices = mapped_indices[(mapped_indices >= start_index) & (mapped_indices < end_index)]
        mapped_indices = mapped_indices[mapped_indices < len(waveform)]  
        label_count = len(mapped_indices)
        label_proportions[label] = label_count / total_points_in_window * 100

    return label_proportions

p_wave_label_proportions = calculate_label_proportions(p_time_window, p_wave_start_index, p_wave_end_index, label_indices, index_mapping)
s_wave_label_proportions = calculate_label_proportions(s_time_window, s_wave_start_index, s_wave_end_index, label_indices, index_mapping)

print("P-wave label proportions:")
for label, proportion in p_wave_label_proportions.items():
    print(f"Label {label + 1}: {proportion:.4f}%")

print("\nS-wave label proportions:")
for label, proportion in s_wave_label_proportions.items():
    print(f"Label {label + 1}: {proportion:.4f}%")

colors = ['#9467bd', '#2ca02c', '#ff69b4', '#d62728', '#ff7f0e']  

plt.figure(figsize=(10, 6))
for label in np.unique(label_indices):
    label_mask = (label_indices == label)
    mapped_indices = index_mapping[label_mask]
   
    p_indices = mapped_indices[(mapped_indices >= p_wave_start_index) & (mapped_indices < p_wave_end_index)]
    p_indices = p_indices[p_indices < len(waveform)]
    plt.plot(p_indices, waveform[p_indices], color=colors[label], alpha=0.6, label=f'Label {label + 1} (P-wave)')
    
    s_indices = mapped_indices[(mapped_indices >= s_wave_start_index) & (mapped_indices < s_wave_end_index)]
    s_indices = s_indices[s_indices < len(waveform)]
    plt.plot(s_indices, waveform[s_indices], color=colors[label], linestyle='--', alpha=0.6, label=f'Label {label + 1} (S-wave)')

plt.axvline(x=p_wave_start_index, color='green', linestyle='--', label=f'P-wave Start ({p_wave_start_time:.2f}s)')
plt.axvline(x=s_wave_start_index, color='red', linestyle='--', label=f'S-wave Start ({s_wave_start_time:.2f}s)')

plt.xlabel('Sampling Point')
plt.ylabel('Amplitude ($10^6$ units)')
plt.title(f'Waveform Snapshot for Station {station} (P-wave and S-wave)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# Labels waveform _ P waves and S waves _ KIGH
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

station = 'KIGH'  
waveform = selected_data[station][0].flatten() * 1e6  
index_mapping = df_sample['index'].values  
label_indices = labels  

fs = 100 


p_wave_threshold = 0.05  
p_wave_start_index = np.argmax(np.abs(waveform) > p_wave_threshold)
p_wave_start_time = p_wave_start_index / fs

s_wave_start_index = p_wave_start_index + int(5 * fs) 
s_wave_threshold = 0.1  
s_wave_start_index = s_wave_start_index + np.argmax(np.abs(waveform[s_wave_start_index:]) > s_wave_threshold)
s_wave_start_time = s_wave_start_index / fs

print(f"P-wave starts at {p_wave_start_time} seconds")
print(f"S-wave starts at {s_wave_start_time} seconds")

window_duration = 3 
p_wave_start_index = max(0, p_wave_start_index)
p_wave_end_index = min(len(waveform), p_wave_start_index + int(window_duration * fs))

s_wave_start_index = max(0, s_wave_start_index)
s_wave_end_index = min(len(waveform), s_wave_start_index + int(window_duration * fs))

waveform_smooth = gaussian_filter1d(waveform, sigma=2) 

p_time_window = np.arange(p_wave_start_index, p_wave_end_index)
s_time_window = np.arange(s_wave_start_index, s_wave_end_index)

def calculate_label_proportions(time_window, start_index, end_index, label_indices, index_mapping):
    total_points_in_window = len(time_window)
    label_proportions = {}

    for label in np.unique(label_indices):
        label_mask = (label_indices == label)
        mapped_indices = index_mapping[label_mask]
        mapped_indices = mapped_indices[(mapped_indices >= start_index) & (mapped_indices < end_index)]
        mapped_indices = mapped_indices[mapped_indices < len(waveform)]  
        label_count = len(mapped_indices)
        label_proportions[label] = label_count / total_points_in_window * 100

    return label_proportions

p_wave_label_proportions = calculate_label_proportions(p_time_window, p_wave_start_index, p_wave_end_index, label_indices, index_mapping)
s_wave_label_proportions = calculate_label_proportions(s_time_window, s_wave_start_index, s_wave_end_index, label_indices, index_mapping)

print("P-wave label proportions:")
for label, proportion in p_wave_label_proportions.items():
    print(f"Label {label + 1}: {proportion:.4f}%")

print("\nS-wave label proportions:")
for label, proportion in s_wave_label_proportions.items():
    print(f"Label {label + 1}: {proportion:.4f}%")

colors = ['#9467bd', '#2ca02c', '#ff69b4', '#d62728', '#ff7f0e']  

plt.figure(figsize=(10, 6))
for label in np.unique(label_indices):
    label_mask = (label_indices == label)
    mapped_indices = index_mapping[label_mask]
   
    p_indices = mapped_indices[(mapped_indices >= p_wave_start_index) & (mapped_indices < p_wave_end_index)]
    p_indices = p_indices[p_indices < len(waveform)]
    plt.plot(p_indices, waveform[p_indices], color=colors[label], alpha=0.6, label=f'Label {label + 1} (P-wave)')
    
    s_indices = mapped_indices[(mapped_indices >= s_wave_start_index) & (mapped_indices < s_wave_end_index)]
    s_indices = s_indices[s_indices < len(waveform)]
    plt.plot(s_indices, waveform[s_indices], color=colors[label], linestyle='--', alpha=0.6, label=f'Label {label + 1} (S-wave)')

plt.axvline(x=p_wave_start_index, color='green', linestyle='--', label=f'P-wave Start ({p_wave_start_time:.2f}s)')
plt.axvline(x=s_wave_start_index, color='red', linestyle='--', label=f'S-wave Start ({s_wave_start_time:.2f}s)')

plt.xlabel('Sampling Point')
plt.ylabel('Amplitude ($10^6$ units)')
plt.title(f'Waveform Snapshot for Station {station} (P-wave and S-wave)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# Labels waveform _ P waves and S waves _ UMWH
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

station = 'UMWH'  
waveform = selected_data[station][0].flatten() * 1e6  
index_mapping = df_sample['index'].values  
label_indices = labels  

fs = 100 


p_wave_threshold = 0.05  
p_wave_start_index = np.argmax(np.abs(waveform) > p_wave_threshold)
p_wave_start_time = p_wave_start_index / fs

s_wave_start_index = p_wave_start_index + int(5 * fs) 
s_wave_threshold = 0.1  
s_wave_start_index = s_wave_start_index + np.argmax(np.abs(waveform[s_wave_start_index:]) > s_wave_threshold)
s_wave_start_time = s_wave_start_index / fs

print(f"P-wave starts at {p_wave_start_time} seconds")
print(f"S-wave starts at {s_wave_start_time} seconds")

window_duration = 3 
p_wave_start_index = max(0, p_wave_start_index)
p_wave_end_index = min(len(waveform), p_wave_start_index + int(window_duration * fs))

s_wave_start_index = max(0, s_wave_start_index)
s_wave_end_index = min(len(waveform), s_wave_start_index + int(window_duration * fs))

waveform_smooth = gaussian_filter1d(waveform, sigma=2) 

p_time_window = np.arange(p_wave_start_index, p_wave_end_index)
s_time_window = np.arange(s_wave_start_index, s_wave_end_index)

def calculate_label_proportions(time_window, start_index, end_index, label_indices, index_mapping):
    total_points_in_window = len(time_window)
    label_proportions = {}

    for label in np.unique(label_indices):
        label_mask = (label_indices == label)
        mapped_indices = index_mapping[label_mask]
        mapped_indices = mapped_indices[(mapped_indices >= start_index) & (mapped_indices < end_index)]
        mapped_indices = mapped_indices[mapped_indices < len(waveform)]  
        label_count = len(mapped_indices)
        label_proportions[label] = label_count / total_points_in_window * 100

    return label_proportions

p_wave_label_proportions = calculate_label_proportions(p_time_window, p_wave_start_index, p_wave_end_index, label_indices, index_mapping)
s_wave_label_proportions = calculate_label_proportions(s_time_window, s_wave_start_index, s_wave_end_index, label_indices, index_mapping)

print("P-wave label proportions:")
for label, proportion in p_wave_label_proportions.items():
    print(f"Label {label + 1}: {proportion:.4f}%")

print("\nS-wave label proportions:")
for label, proportion in s_wave_label_proportions.items():
    print(f"Label {label + 1}: {proportion:.4f}%")

colors = ['#9467bd', '#2ca02c', '#ff69b4', '#d62728', '#ff7f0e']  

plt.figure(figsize=(10, 6))
for label in np.unique(label_indices):
    label_mask = (label_indices == label)
    mapped_indices = index_mapping[label_mask]
   
    p_indices = mapped_indices[(mapped_indices >= p_wave_start_index) & (mapped_indices < p_wave_end_index)]
    p_indices = p_indices[p_indices < len(waveform)]
    plt.plot(p_indices, waveform[p_indices], color=colors[label], alpha=0.6, label=f'Label {label + 1} (P-wave)')
    
    s_indices = mapped_indices[(mapped_indices >= s_wave_start_index) & (mapped_indices < s_wave_end_index)]
    s_indices = s_indices[s_indices < len(waveform)]
    plt.plot(s_indices, waveform[s_indices], color=colors[label], linestyle='--', alpha=0.6, label=f'Label {label + 1} (S-wave)')

plt.axvline(x=p_wave_start_index, color='green', linestyle='--', label=f'P-wave Start ({p_wave_start_time:.2f}s)')
plt.axvline(x=s_wave_start_index, color='red', linestyle='--', label=f'S-wave Start ({s_wave_start_time:.2f}s)')

plt.xlabel('Sampling Point')
plt.ylabel('Amplitude ($10^6$ units)')
plt.title(f'Waveform Snapshot for Station {station} (P-wave and S-wave)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# Labels waveform _ P waves and S waves _ ASKH
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

station = 'ASKH'  
waveform = selected_data[station][0].flatten() * 1e6  
index_mapping = df_sample['index'].values  
label_indices = labels  

fs = 100 


p_wave_threshold = 0.05  
p_wave_start_index = np.argmax(np.abs(waveform) > p_wave_threshold)
p_wave_start_time = p_wave_start_index / fs

s_wave_start_index = p_wave_start_index + int(5 * fs) 
s_wave_threshold = 0.1  
s_wave_start_index = s_wave_start_index + np.argmax(np.abs(waveform[s_wave_start_index:]) > s_wave_threshold)
s_wave_start_time = s_wave_start_index / fs

print(f"P-wave starts at {p_wave_start_time} seconds")
print(f"S-wave starts at {s_wave_start_time} seconds")

window_duration = 3 
p_wave_start_index = max(0, p_wave_start_index)
p_wave_end_index = min(len(waveform), p_wave_start_index + int(window_duration * fs))

s_wave_start_index = max(0, s_wave_start_index)
s_wave_end_index = min(len(waveform), s_wave_start_index + int(window_duration * fs))

waveform_smooth = gaussian_filter1d(waveform, sigma=2) 

p_time_window = np.arange(p_wave_start_index, p_wave_end_index)
s_time_window = np.arange(s_wave_start_index, s_wave_end_index)

def calculate_label_proportions(time_window, start_index, end_index, label_indices, index_mapping):
    total_points_in_window = len(time_window)
    label_proportions = {}

    for label in np.unique(label_indices):
        label_mask = (label_indices == label)
        mapped_indices = index_mapping[label_mask]
        mapped_indices = mapped_indices[(mapped_indices >= start_index) & (mapped_indices < end_index)]
        mapped_indices = mapped_indices[mapped_indices < len(waveform)]  
        label_count = len(mapped_indices)
        label_proportions[label] = label_count / total_points_in_window * 100

    return label_proportions

p_wave_label_proportions = calculate_label_proportions(p_time_window, p_wave_start_index, p_wave_end_index, label_indices, index_mapping)
s_wave_label_proportions = calculate_label_proportions(s_time_window, s_wave_start_index, s_wave_end_index, label_indices, index_mapping)

print("P-wave label proportions:")
for label, proportion in p_wave_label_proportions.items():
    print(f"Label {label + 1}: {proportion:.4f}%")

print("\nS-wave label proportions:")
for label, proportion in s_wave_label_proportions.items():
    print(f"Label {label + 1}: {proportion:.4f}%")

colors = ['#9467bd', '#2ca02c', '#ff69b4', '#d62728', '#ff7f0e']  

plt.figure(figsize=(10, 6))
for label in np.unique(label_indices):
    label_mask = (label_indices == label)
    mapped_indices = index_mapping[label_mask]
   
    p_indices = mapped_indices[(mapped_indices >= p_wave_start_index) & (mapped_indices < p_wave_end_index)]
    p_indices = p_indices[p_indices < len(waveform)]
    plt.plot(p_indices, waveform[p_indices], color=colors[label], alpha=0.6, label=f'Label {label + 1} (P-wave)')
    
    s_indices = mapped_indices[(mapped_indices >= s_wave_start_index) & (mapped_indices < s_wave_end_index)]
    s_indices = s_indices[s_indices < len(waveform)]
    plt.plot(s_indices, waveform[s_indices], color=colors[label], linestyle='--', alpha=0.6, label=f'Label {label + 1} (S-wave)')

plt.axvline(x=p_wave_start_index, color='green', linestyle='--', label=f'P-wave Start ({p_wave_start_time:.2f}s)')
plt.axvline(x=s_wave_start_index, color='red', linestyle='--', label=f'S-wave Start ({s_wave_start_time:.2f}s)')

plt.xlabel('Sampling Point')
plt.ylabel('Amplitude ($10^6$ units)')
plt.title(f'Waveform Snapshot for Station {station} (P-wave and S-wave)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# Save centroid
non_earthquake_centroids = km_full.cluster_centers_
print("Non Earthquake Centroids (Shape: {}):".format(non_earthquake_centroids.shape))
print(non_earthquake_centroids)
np.save('non_earthquake_centroids.npy', non_earthquake_centroids)  


# In[ ]:


# Calculate the distance between the centroid and the
import numpy as np
from scipy.spatial.distance import cdist

earthquake_centroids = np.load('earthquake_centroids.npy')
non_earthquake_centroids = np.load('non_earthquake_centroids.npy')
print(f"Earthquake Centroids (Shape: {earthquake_centroids.shape}):\n", earthquake_centroids)
print(f"Non Earthquake Centroids (Shape: {non_earthquake_centroids.shape}):\n", non_earthquake_centroids)

distance_matrix = cdist(earthquake_centroids, non_earthquake_centroids, metric='euclidean')
print("Distance Matrix between Earthquake and Non-Earthquake Centroids:\n", distance_matrix)

min_distance = np.min(distance_matrix)
max_distance = np.max(distance_matrix)
mean_distance = np.mean(distance_matrix)

print(f"Minimum distance between centroids: {min_distance}")
print(f"Maximum distance between centroids: {max_distance}")
print(f"Mean distance between centroids: {mean_distance}")


# In[ ]:


# Label time series comparison
import matplotlib.pyplot as plt
import numpy as np

earthquake_time_series_labels = np.load('earthquake_time_series_labels.npy')
non_earthquake_time_series_labels = np.load('non_earthquake_time_series_labels.npy')

total_intervals_earthquake = earthquake_time_series_labels.shape[0]
total_intervals_non_earthquake = non_earthquake_time_series_labels.shape[0]
num_clusters = earthquake_time_series_labels.shape[1]
colors = ['#9467bd', '#2ca02c', '#ff69b4', '#d62728', '#ff7f0e'] 

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
for cluster_id in range(num_clusters):
    plt.plot(range(total_intervals_earthquake), earthquake_time_series_labels[:, cluster_id], 
             color=colors[cluster_id], label=f'Label {cluster_id + 1}')
plt.xlabel('Time Intervals (Earthquake Group)')
plt.ylabel('Percentage of Cluster Labels')
plt.title('Cluster Label Distribution Over Time - Earthquake Group')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
for cluster_id in range(num_clusters):
    plt.plot(range(total_intervals_non_earthquake), non_earthquake_time_series_labels[:, cluster_id], 
             color=colors[cluster_id], label=f'Label {cluster_id + 1}')
plt.xlabel('Time Intervals (Non-Earthquake Group)')
plt.ylabel('Percentage of Cluster Labels')
plt.title('Cluster Label Distribution Over Time - Non-Earthquake Group')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[ ]:


# Z-score
import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Mean of the earthquake period (X)': [11.48, 13.25, 33.61, 8.87, 32.78],
    'Mean of the 12-hour period (μ)': [8.83, 14.42, 35.05, 18.94, 22.76],
    'Standard deviation of the 12-hour period (σ)': [0.11, 0.13, 0.13, 0.14, 0.14],
    'Z-score': [0.25, -0.09, -0.11, -0.71, 0.72]
}

df = pd.DataFrame(data, index=[f'Label {i+1}' for i in range(5)])

styled_df = df.style.set_properties(**{
    'font-size': '10pt',  
    'text-align': 'center'  
}).set_table_styles([{
    'selector': 'th',
    'props': [('font-size', '10pt'), ('text-align', 'center')]  
}])

styled_df

