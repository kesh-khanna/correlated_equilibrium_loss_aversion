"""
Clean and analyze the experimental data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filename):
    """
    load in the data and organize the columns
    :return:
    """
    data = pd.read_csv(filename)

    # combine Recommendation 1 - 10 with Participant Action 1 - 10 as a round 1 - 10 tuple
    for i in range(1, 11):
        recommendation_col = "Recommendation " + str(i)
        action_col = "Participant Response " + str(i)

        # change from A and S to 0 and 1
        data[recommendation_col] = data[recommendation_col].apply(lambda x: 0 if x == "A" else 1)
        data[action_col] = data[action_col].apply(lambda x: 0 if x == "A" else 1)

        data["Round " + str(i)] = list(zip(data[recommendation_col], data[action_col]))

        # drop the original columns
        data.drop(columns=[recommendation_col, action_col], inplace=True)

    return data


def confusion_matrix(filename):
    """
    Create a confusion matrix for the data where the rows are the recommended action and
    the column is the actual action taken.
    :return:
    """
    # load and organize the data
    data = load_data(filename)

    # split the data into two groups. Game A and B
    game_a = data[data["Game"] == "A"]
    game_b = data[data["Game"] == "B"]

    # create a confusion matrix for each game
    confusion_matrix_a = np.zeros((2, 2))
    confusion_matrix_b = np.zeros((2, 2))

    # for each round in the game update the confusion matrix
    for i in range(1, 11):
        round_col = "Round " + str(i)
        for round_data in game_a[round_col]:
            recommendation, action = round_data
            confusion_matrix_a[recommendation][action] += 1

        for round_data in game_b[round_col]:
            recommendation, action = round_data
            confusion_matrix_b[recommendation][action] += 1

    # plot the confusion matrix
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(confusion_matrix_a, annot=True, ax=ax[0], cmap="Blues", fmt='g')
    ax[0].set_title("Game A (Non-Loss)")
    ax[0].set_xlabel("Actual Action")
    ax[0].set_ylabel("Recommended Action")

    sns.heatmap(confusion_matrix_b, annot=True, ax=ax[1], cmap="Blues", fmt='g')
    ax[1].set_title("Game B (Loss)")
    ax[1].set_xlabel("Actual Action")
    ax[1].set_ylabel("Recommended Action")

    # Change labels back from 0 and 1 to A and S
    labels = ["A", "S"]
    ax[0].set_xticklabels(labels)
    ax[0].set_yticklabels(labels)
    ax[1].set_xticklabels(labels)
    ax[1].set_yticklabels(labels)

    # increase label size
    ax[0].tick_params(axis='both', which='major', labelsize=12)
    ax[1].tick_params(axis='both', which='major', labelsize=12)


    plt.savefig("confusion_matrix.png")

def confusion_matrix_by_round(filename):
    """
    Create a confusion matrix for the data where the rows are the recommended action and
    the column is the actual action taken.
    :return:
    """
    # load and organize the data
    data = load_data(filename)

    # split the data into two groups. Game A and B
    game_a = data[data["Game"] == "A"]
    game_b = data[data["Game"] == "B"]

    # create a confusion matrix for each game
    confusion_matrix_a = np.zeros((2, 2, 10))
    confusion_matrix_b = np.zeros((2, 2, 10))

    # for each round in the game update the confusion matrix
    for i in range(1, 11):
        round_col = "Round " + str(i)
        for round_data in game_a[round_col]:
            recommendation, action = round_data
            confusion_matrix_a[recommendation][action][i - 1] += 1

        for round_data in game_b[round_col]:
            recommendation, action = round_data
            confusion_matrix_b[recommendation][action][i - 1] += 1

    # plot the confusion matrix
    fig, ax = plt.subplots(2, 10, figsize=(24, 12))
    for i in range(10):
        sns.heatmap(confusion_matrix_a[:, :, i], annot=True, ax=ax[0, i], cmap="Blues", fmt='g')
        ax[0, i].set_title("Game A Round " + str(i + 1))
        ax[0, i].set_xlabel("Actual Action")
        ax[0, i].set_ylabel("Recommended Action")

        sns.heatmap(confusion_matrix_b[:, :, i], annot=True, ax=ax[1, i], cmap="Blues", fmt='g')
        ax[1, i].set_title("Game B Round " + str(i + 1))
        ax[1, i].set_xlabel("Actual Action")
        ax[1, i].set_ylabel("Recommended Action")

    # Change labels back from 0 and 1 to A and S
    labels = ["A", "S"]
    for i in range(10):
        ax[0, i].set_xticklabels(labels)
        ax[0, i].set_yticklabels(labels)
        ax[1, i].set_xticklabels(labels)
        ax[1, i].set_yticklabels(labels)

    plt.savefig("confusion_matrix_by_round.png")

def confusion_matrix_percentage(filename):
    """
    Same as orginal confusion matrix but with percentages instead of raw counds
    :param filename:
    :return:
    """
    # load and organize the data
    data = load_data(filename)

    # split the data into two groups. Game A and B
    game_a = data[data["Game"] == "A"]
    game_b = data[data["Game"] == "B"]

    # create a confusion matrix for each game
    confusion_matrix_a = np.zeros((2, 2))
    confusion_matrix_b = np.zeros((2, 2))

    # for each round in the game update the confusion matrix
    for i in range(1, 11):
        round_col = "Round " + str(i)
        for round_data in game_a[round_col]:
            recommendation, action = round_data
            confusion_matrix_a[recommendation][action] += 1

        for round_data in game_b[round_col]:
            recommendation, action = round_data
            confusion_matrix_b[recommendation][action] += 1

    # convert to percentages
    confusion_matrix_a = confusion_matrix_a / confusion_matrix_a.sum(axis=1)[:, None]
    confusion_matrix_b = confusion_matrix_b / confusion_matrix_b.sum(axis=1)[:, None]

    # plot the confusion matrix
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(confusion_matrix_a, annot=True, ax=ax[0], cmap="Blues", fmt='.2f')
    ax[0].set_title("Game A (Non-Loss)")
    ax[0].set_xlabel("Actual Action")
    ax[0].set_ylabel("Recommended Action")

    sns.heatmap(confusion_matrix_b, annot=True, ax=ax[1], cmap="Blues", fmt='.2f')
    ax[1].set_title("Game B (Loss)")
    ax[1].set_xlabel("Actual Action")
    ax[1].set_ylabel("Recommended Action")

    # Change labels back from 0 and 1 to A and S
    labels = ["A", "S"]
    ax[0].set_xticklabels(labels)
    ax[0].set_yticklabels(labels)
    ax[1].set_xticklabels(labels)
    ax[1].set_yticklabels(labels)

    # increase label size
    ax[0].tick_params(axis='both', which='major', labelsize=12)
    ax[1].tick_params(axis='both', which='major', labelsize=12)

    plt.savefig("confusion_matrix_percentage.png")


def get_statistics(filename):
    """
    Calculate a few important statistics for this game
    :return:
    """
    data = load_data(filename)

    # split the data into two groups. Game A and B
    game_a = data[data["Game"] == "A"]
    game_b = data[data["Game"] == "B"]

    comply_a = 0
    comply_b = 0
    total_a = 0
    total_b = 0

    # calculate the number of times that the recommendation was followed for each game
    for i in range(1, 11):
        round_col = "Round " + str(i)
        for round_data in game_a[round_col]:
            recommendation, action = round_data

            if action == recommendation:
                comply_a += 1

            total_a += 1

        for round_data in game_b[round_col]:
            recommendation, action = round_data

            if action == recommendation:
                comply_b += 1

            total_b += 1

    print("Total compliance in Game A: " + str(comply_a))
    print("Total compliance in Game B: " + str(comply_b))
    print("Total A: " + str(total_a))
    print("Total B: " + str(total_b))

    print("Game A Comply %: " + str(round(comply_a / total_a, 2)))
    print("Game B Comply %: " + str(round(comply_b / total_b, 2)))

    # now look at the compliance conditional on the recommendation
    # compliance when recommended A vs compliance when recommended B
    comply_a_a = 0
    comply_a_b = 0
    comply_b_a = 0
    comply_b_b = 0

    total_a_a = 0
    total_a_b = 0
    total_b_a = 0
    total_b_b = 0

    for i in range(1, 11):
        round_col = "Round " + str(i)
        for round_data in game_a[round_col]:
            recommendation, action = round_data

            if recommendation == 0:
                total_a_a += 1
                if action == recommendation:
                    comply_a_a += 1
            else:
                total_a_b += 1
                if action == recommendation:
                    comply_a_b += 1

        for round_data in game_b[round_col]:
            recommendation, action = round_data

            if recommendation == 0:
                total_b_a += 1
                if action == recommendation:
                    comply_b_a += 1
            else:
                total_b_b += 1
                if action == recommendation:
                    comply_b_b += 1

    print("Game A Comply % when recommended A: " + str(round(comply_a_a / total_a_a, 2)))
    print("Game A Comply % when recommended B: " + str(round(comply_a_b / total_a_b, 2)))
    print("Game B Comply % when recommended A: " + str(round(comply_b_a / total_b_a, 2)))
    print("Game B Comply % when recommended B: " + str(round(comply_b_b / total_b_b, 2)))


def game_a_outcome(filename):
    """
    By pairing row players with a column player from game A we can create a heatmap that shows the average outcome
    :param filename:
    :return:
    """
    data = load_data(filename)

    game_a = data[data["Game"] == "A"]

    outcome_matrix = np.zeros((2, 2))
    theory_matrix = np.zeros((2, 2))

    game_a_row = game_a[game_a["Player Type"] == "Row"]
    game_a_col = game_a[game_a["Player Type"] == "Column"]

    assert len(game_a_row) == len(game_a_col)

    # shuffle the data so that the row and column players are paired randomly
    game_a_row = game_a_row.sample(frac=1)
    game_a_col = game_a_col.sample(frac=1)

    for i in range(len(game_a_row)):
        row_data = game_a_row.iloc[i]
        col_data = game_a_col.iloc[i]

        for i in range(1, 11):
            round_col = "Round " + str(i)
            round_data = row_data[round_col]
            rec_row, action_row = round_data

            round_data = col_data[round_col]
            rec_col, action_col = round_data

            outcome_matrix[action_row][action_col] += 1

            # calculate the theoretical outcome based on the recommendation
            theory_matrix[rec_row][rec_col] += 1


    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.heatmap(outcome_matrix, annot=True, ax=ax, cmap="Reds", fmt='g')
    ax.set_title("Game A Outcome")
    ax.set_xlabel("Column Player Action")
    ax.set_ylabel("Row Player Action")

    labels = ["A", "S"]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # fig title
    plt.title("Game A Outcome")
    plt.savefig("game_a_outcome.png")

    # save the theoretical matrix
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.heatmap(theory_matrix, annot=True, ax=ax, cmap="Reds", fmt='g')
    ax.set_title("Game A Theoretical Outcome")
    ax.set_xlabel("Column Player Recommendation")
    ax.set_ylabel("Row Player Recommendation")

    labels = ["A", "S"]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # fig title
    plt.title("Game A Theoretical Outcome")
    plt.savefig("game_a_theoretical_outcome.png")


def statistics_by_round(filename):
    """
    Look at the same statistics from get_statistics and how they change per round
    :return:
    """
    # load and organize the data
    data = load_data(filename)

    # split the data into two groups. Game A and B
    game_a = data[data["Game"] == "A"]
    game_b = data[data["Game"] == "B"]

    comply_a = 0
    comply_b = 0
    total_a = 0
    total_b = 0

    round_comply_a = []
    round_comply_b = []

    # calculate the number of times that the recommendation was followed for each game in each round
    for i in range(1, 11):
        round_col = "Round " + str(i)
        for round_data in game_a[round_col]:
            recommendation, action = round_data

            if action == recommendation:
                comply_a += 1

            total_a += 1

        for round_data in game_b[round_col]:
            recommendation, action = round_data

            if action == recommendation:
                comply_b += 1

            total_b += 1

        print("Round " + str(i) + " Compliance in Game A: " + str(comply_a))
        print("Round " + str(i) + " Compliance in Game B: " + str(comply_b))
        print("Round " + str(i) + " Total A: " + str(total_a))
        print("Round " + str(i) + " Total B: " + str(total_b))

        print("Round " + str(i) + " Game A Comply %: " + str(round(comply_a / total_a, 2)))
        print("Round " + str(i) + " Game B Comply %: " + str(round(comply_b / total_b, 2)))

        round_comply_a.append(comply_a / total_a)
        round_comply_b.append(comply_b / total_b)

        comply_a = 0
        comply_b = 0
        total_a = 0
        total_b = 0

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(range(1, 11), round_comply_a, label="Game A (Non-Loss)")
    ax.plot(range(1, 11), round_comply_b, label="Game B (Loss)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Compliance %")
    ax.legend()
    plt.savefig("statistics_by_round.png")


def loss_aversion_stats(filename):
    """
    Take a look at the survey reponse, examine basic stats then see if there
    is any correlation with their actions in the game
    :param filename:
    :return:
    """
    data = load_data(filename)

    # change columns to be more readable
    data.rename(columns={"Consider Opponents Loss Aversion?": "consider",
                         "Comparison of Loss Aversion": "comparison"}, inplace=True)

    # if the participant reponded no to considering the opponents loss aversion then
    # they should have n/a for comparison of loss aversion
    data.loc[data["consider"] == "No", "comparison"] = None

    considered = data["consider"].value_counts()
    comparison = data["comparison"].value_counts()

    print(considered)

    print(comparison)

    # now look at the correlation between the survey response and the actions in the game

    # split data into yes and no groups
    data_yes = data[data["consider"] == "Yes"]
    data_no = data[data["consider"] == "No"]

    # calculate the compliance rate for each group
    comply_yes = 0
    comply_no = 0
    total_yes = 0
    total_no = 0

    for i in range(1, 11):
        round_col = "Round " + str(i)
        for round_data in data_yes[round_col]:
            recommendation, action = round_data

            if action == recommendation:
                comply_yes += 1

            total_yes += 1

        for round_data in data_no[round_col]:
            recommendation, action = round_data

            if action == recommendation:
                comply_no += 1

            total_no += 1

    print("Compliance rate for those who considered opponents loss aversion: " + str(comply_yes / total_yes))
    print(total_yes)

    print("Compliance rate for those who did not consider opponents loss aversion: " + str(comply_no / total_no))
    print(total_no)

    # check the conditional compliance rate
    comply_yes_a = 0
    comply_yes_b = 0
    comply_no_a = 0
    comply_no_b = 0

    total_yes_a = 0
    total_yes_b = 0
    total_no_a = 0
    total_no_b = 0

    for i in range(1, 11):
        round_col = "Round " + str(i)
        for round_data in data_yes[round_col]:
            recommendation, action = round_data

            if recommendation == 0:
                total_yes_a += 1
                if action == recommendation:
                    comply_yes_a += 1
            else:
                total_yes_b += 1
                if action == recommendation:
                    comply_yes_b += 1

        for round_data in data_no[round_col]:
            recommendation, action = round_data

            if recommendation == 0:
                total_no_a += 1
                if action == recommendation:
                    comply_no_a += 1
            else:
                total_no_b += 1
                if action == recommendation:
                    comply_no_b += 1

    print("Compliance rate for those who considered opponents loss aversion when recommended A: " + str(comply_yes_a / total_yes_a))
    print(total_yes_a)
    print("Compliance rate for those who considered opponents loss aversion when recommended S: " + str(comply_yes_b / total_yes_b))
    print(total_yes_b)
    print("Compliance rate for those who did not consider opponents loss aversion when recommended A: " + str(comply_no_a / total_no_a))
    print(total_no_a)
    print("Compliance rate for those who did not consider opponents loss aversion when recommended S: " + str(comply_no_b / total_no_b))
    print(total_no_b)


if __name__ == "__main__":
    # confusion_matrix("data.csv")
    # confusion_matrix_by_round("data.csv")
    # confusion_matrix_percentage("data.csv")
    # get_statistics("data.csv")
    # game_a_outcome("data.csv")
    # statistics_by_round("data.csv")

    loss_aversion_stats("data.csv")

