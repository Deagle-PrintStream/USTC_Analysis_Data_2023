import numpy as np
import pickle
import matplotlib.pyplot as plt


def pred_distribution(y_true, y_pred, output_layer_size: int = 3) -> np.ndarray:
    correct_mat = np.zeros((output_layer_size, output_layer_size))
    for i in range(0, len(y_true)):
        true_tag = y_true[i]
        pred_tag = y_pred[i]
        correct_mat[true_tag][pred_tag] += 1

    for i in range(0, output_layer_size):
        correct_mat[i] /= sum(correct_mat[i])

    print("prediction distribution:\n", correct_mat)
    return correct_mat


def save_model(myModel, fout_path: str) -> None:
    obj = pickle.dumps(myModel)
    with open(fout_path, "wb") as f_out:
        f_out.write(obj)
        f_out.close()


def load_model(obj_path: str):
    with open(obj_path, "rb") as f_in:
        obj = pickle.load(f_in)
        f_in.close()
    return obj


def test_model(model, x_valid, y_valid) -> float:
    y_pred = model.predict(x_valid)
    acc = round(np.sum(y_valid == y_pred) / len(y_valid), 2)
    print("acc={:.2f}%\n".format(100 * acc))
    pred_distribution(y_valid, y_pred, 3)
    return acc


def plot_learning_curve(loss_list: list) -> None:
    plt.figure(figsize=(10, 15), dpi=100)
    plt.xlabel("epochs", fontdict={"size": 16})
    plt.ylabel("loss", fontdict={"size": 16})
    plt.title("Learning curve", fontdict={"size": 18})
    x = [e[0] for e in loss_list]
    y = [c[1] for c in loss_list]
    plt.plot(x, y)
    plt.show()
