import numpy as np

# training_set = np.loadtxt('train_100k_withratings.csv', delimiter=',')
# testing_set = np.loadtxt('test_100k_withoutratings.csv', delimiter=',')

# example from slides
example_train = np.array([[1,1,8],               [1,4,2],[1,5,7],
                          [2,1,2],        [2,3,5],[2,4,7],[2,5,5],
                          [3,1,5],[3,2,4],[3,3,7],[3,4,4],[3,5,7],
                          [4,1,7],[4,2,1],[4,3,7],[4,4,3],[4,5,8],
                          [5,1,1],[5,2,7],[5,3,4],[5,4,6],[5,5,5],
                          [6,1,8],[6,2,3],[6,3,8],[6,4,3],[6,5,7]])
example_test = np.array([[1,2,21],[1,3,22],[2,2,23]])
training_set = example_train
testing_set = example_test

print("training set: \n", training_set)

# arrays of users, items, and their max value
items = np.unique(training_set[:, 1])
max_item = items.max()
users = np.unique(training_set[:, 0])
max_user = users.max()
print(max_user)

# create array of user averages
user_totals = np.zeros(max_user)
user_count = np.zeros(max_user)
for rating in training_set:
    user_totals[rating[0]-1] += rating[2]   #index 0 is user 1 etc.
    user_count[rating[0]-1] += 1
user_averages = user_totals / user_count
print("user averages:\n", user_averages)

# create user item ratings matrix
ratings_matrix = np.zeros((max_user, max_item))
for rating in training_set:
    ratings_matrix[rating[0]-1][rating[1]-1] = rating[2]
print("user item ratings matrix:\n", ratings_matrix)

# create average (user) subtracted matrix
avg_sub_matrix = np.zeros((max_user, max_item))
for u in range(max_user):
    avg_sub_matrix[u] = np.where(ratings_matrix[u] > 0, ratings_matrix[u] - user_averages[u], 0) #subtract user average from every rating except 0 (missing) ratings
print("average subtracted matrix:\n", avg_sub_matrix)

# create item similarity matrix
item_sim_matrix = np.zeros((max_item, max_item))
for i1 in range(max_item):
    for i2 in range(max_item):  #every item computed with every other item
        item1 = np.where(ratings_matrix[:, i2] != 0, avg_sub_matrix[:, i1], 0)  #if a user hasn't rated either item, both ratings are set to 0 to be excluded from the calculation
        item2 = np.where(ratings_matrix[:, i1] != 0, avg_sub_matrix[:, i2], 0)
        # calculate adjusted cosine similarity
        numerator = sum(item1 * item2)
        denominator = np.sqrt(sum(item1 ** 2)) * np.sqrt(sum(item2 ** 2))
        similarity = numerator / denominator
        item_sim_matrix[i1][i2] = similarity
print("item similarity matrix:\n", item_sim_matrix)

print("testing set:\n", testing_set)

# predict ratings
results = np.array([])
for user_item in testing_set:
    item_sim = np.where((item_sim_matrix[user_item[1]-1] > 0), item_sim_matrix[user_item[1]-1], 0)  #neighbourhood is items with positive similarity
    item_sim[user_item[1]-1] = 0    #item being predicted (similarity of 1) set to 0 to be excluded from the calculation
    # prediction calculation
    numerator = sum(item_sim * avg_sub_matrix[user_item[0]-1])
    denominator = sum(item_sim)
    prediction = user_averages[user_item[0]-1] + (numerator / denominator)
    # prediction rounding
    if prediction < 0.5:    #lowest rating is 0.5
        prediction = 0.5
    elif prediction > 8:    #highest rating is 5
        prediction = 5
    else:
        prediction = (round(prediction * 2)) / 2    #round to nearest 0.5
    results = np.append(results, prediction)

split_testing_set = np.hsplit(testing_set, [2]) #split testing set to separate user/items and timestamp
results_set = np.hstack((split_testing_set[0], np.atleast_2d(results).T))   #insert results after user/items and timestamp
results_set = np.hstack((results_set, split_testing_set[1]))
print("results set:\n", results_set)

# # write to file 
# np.savetxt('results.csv', results_set, delimiter=',')

