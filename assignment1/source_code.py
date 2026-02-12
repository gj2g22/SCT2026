import numpy as np

train = np.loadtxt('train_100k_withratings.csv', delimiter=',')
print(train)
print(np.shape(train))

# example from slides
example = np.array([[1,1,8],               [1,4,2],[1,5,7],
                   [2,1,2],        [2,3,5],[2,4,7],[2,5,5],
                   [3,1,5],[3,2,4],[3,3,7],[3,4,4],[3,5,7],
                   [4,1,7],[4,2,1],[4,3,7],[4,4,3],[4,5,8],
                   [5,1,1],[5,2,7],[5,3,4],[5,4,6],[5,5,5],
                   [6,1,8],[6,2,3],[6,3,8],[6,4,3],[6,5,7]])

# arrays of users, items, and their max value
items = np.unique(example[:, 1])
max_item = items.max()
users = np.unique(example[:, 0])
max_user = users.max()

# create array of user averages
user_totals = np.zeros(max_user)
user_count = np.zeros(max_user)
for rating in example:
    user_totals[rating[0]-1] += rating[2]   #index 0 is user 1 etc.
    user_count[rating[0]-1] += 1
user_averages = user_totals / user_count
print("user averages:\n", user_averages)

# create user item ratings matrix
user_item_matrix = np.zeros((max_user, max_item))
print("user item matrix:\n", user_item_matrix)

# create item similarity matrix
item_sim_matrix = np.zeros((max_item, max_item))
print("item similarity matrix:\n", item_sim_matrix)

