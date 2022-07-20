import components
import constants
import heapq
from collections import defaultdict
from operator import itemgetter

#building a trainset from the dataset
dataset = components.ml_small_rating_to_dataset()
trainset = dataset.build_full_trainset()


#loading the trained model (matrix)
model_filename = constants.KNNBASICMODEL
similarity_matrix = components.load_model(model_filename)

user_id = int(input("enter the user id \n"))

#calculating by using 20 nearest neighbors
k = 20

#finding the top 20 rated movies by user
test_subject_IID = trainset.to_inner_uid(user_id)
test_subject_ratings = trainset.ur[test_subject_IID]
k_neighbours = heapq.nlargest(k, test_subject_ratings, key= lambda x: x[1])

#will thrwo keyerror if we use a normal dictionary since we cannot search with a non-existent key in a normal dict
#finding similarities of each element in k_neighbours and storing them by assigning each of them a score
#to improvde the accuracy of the score modifying the default score as score*(rating/5.0) 
candidates = defaultdict(float)

for itemID , rating in k_neighbours:
    similarities = similarity_matrix[itemID]
    for innerID, score in enumerate(similarities):
        candidates[innerID] += score*(rating / 5.0)

watched = []
for itemID, rating in trainset.ur[test_subject_IID]:
    watched.append(itemID)

recommendation = []

position = 0


#candidates have a structure of innerid : score hence we need to sort candidates descending order of score
for itemID,_ in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in watched:
        recommendation.append(components.movieid_to_name(trainset.to_raw_iid(itemID)))
        position+=1
        if ( position > 10) : break

for rec in recommendation:
    print("Movie :", rec)





