#!/usr/bin/env python3
"""Imports"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid,
                     Y_valid, batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Its Trainin Time"""
    saver = tf.train.import_meta_graph(load_path + '.meta')

    with tf.Session() as sesh:
        saver.restore(sesh, load_path)

        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name('x:0')
        y = graph.get_tensor_by_name('y:0')

        accuracy = graph.get_tensor_by_name('accuracy:0')
        loss = graph.get_tensor_by_name('loss:0')
        train_op = graph.get_operation_by_name('train_op')

        m_train = X_train.shape[0]
        num_batches = int(np.ceil(m_train / batch_size))

        for i in range(epochs + 1):
            print("After {} epochs:".format(i))

            X_train, Y_train = shuffle_data(X_train, Y_train)

            total_cost = 0.0
            total_accuracy = 0.0

            for j in range(num_batches):
                start_ind = j * batch_size
                end_ind = min((j + 1) * batch_size, m_train)
                X_batch = X_train[start_ind:end_ind]
                Y_batch = Y_train[start_ind:end_ind]

                _, batch_cost, batch_accuracy = sesh.run(
                    [train_op, loss, accuracy],
                    feed_dict={x: X_batch, y: Y_batch}
                )

                total_cost += batch_cost
                total_accuracy += batch_accuracy

                if (j + 1) % 100 == 0:
                    print("\tStep {}:".format(j + 1))
                    print("\t\tCost: {}".format(batch_cost))
                    print("\t\tAccuracy: {}".format(batch_accuracy))

            avg_cost = total_cost / num_batches
            avg_accuracy = total_accuracy / num_batches

            valid_cost, valid_accuracy = sesh.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid}
            )

            print("\tTraining Cost: {}".format(avg_cost))
            print("\tTraining Accuracy: {}".format(avg_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

        save_path = saver.save(sesh, save_path)
        print("Model saved in path: {}".format(save_path))

    return save_path
