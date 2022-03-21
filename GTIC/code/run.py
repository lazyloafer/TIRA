import MV_feature
import CRH
import readFile
import kmeans_xjz
import time


# dataset = "d_Duck_Identification"
# dataset = "WeatherSentiment"
# dataset = "SP"
# dataset = "s4_Dog_data"
# dataset = "d_jn-product"
# dataset = "HITspam-UsingCrowdflower"
# dataset = "valence7"
# dataset = "valence5"
# dataset = "aircrowd6"
# dataset = "CF"
# dataset = "fej2013"
dataset = "trec2011"


print("GTIC:")
time_start = time.time()
MV_feature.generate_MV_feature(dataset)
kmeans_xjz.kmeans(dataset, "MV_feature")
time_end = time.time()
time_run = time_end - time_start
print('timecost:', time_run, 's')
