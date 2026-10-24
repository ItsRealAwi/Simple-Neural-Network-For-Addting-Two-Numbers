import decimal
import time
import numpy as np

decimal.getcontext().prec = 18

ZERO = decimal.Decimal("0")
ONE = decimal.Decimal("1")
TWO = decimal.Decimal("2")


def ReLU(x):
    return ZERO.max(x)


def ReLU_derivative(x):
    return ONE if x > ZERO else ZERO



weights_1 = np.random.uniform(low=-0.05, high=0.05, size=(32, 20))
weights_2 = np.random.uniform(low=-0.05, high=0.05, size=(19, 32))
bias_1 = np.random.uniform(low=-0.05, high=0.05, size=32)
bias_2 = np.random.uniform(low=-0.05, high=0.05, size=19)


dataset = []
for a in range(10):
    for b in range(10):
        inp_vec = [decimal.Decimal(str(val)) for val in np.array([0] * 20, np.int8)]
        inp_vec[a] = ONE
        inp_vec[10 + b] = ONE

        tgt_sum = a + b
        tgt_vec = [ZERO] * 19
        tgt_vec[tgt_sum] = ONE

        dataset.append((inp_vec, tgt_vec, a, b, tgt_sum))

learning_rate = decimal.Decimal("0.005")
epochs = 300

print(
    f"\n[START OF OPTIMIZED TRAINING] 100 combinations, epochs: {epochs}...\n"
)
start_time = time.time()

for epoch in range(epochs):
    total_epoch_loss = ZERO

    for input_vector, target_vector, a, b, target_output in dataset:
        hidden_layer_pre_activation = []
        hidden_layer_outputs = []

        for i in range(32):
            neuron_sum = decimal.Decimal(str(bias_1[i]))
            w_row = weights_1[i]
            for j in range(20):
                neuron_sum += decimal.Decimal(str(w_row[j])) * input_vector[j]

            hidden_layer_pre_activation.append(neuron_sum)
            hidden_layer_outputs.append(ReLU(neuron_sum))

        output_layer_logits = []
        for i in range(19):
            neuron_sum = decimal.Decimal(str(bias_2[i]))
            w_row = weights_2[i]
            for j in range(32):
                neuron_sum += decimal.Decimal(str(w_row[j])) * hidden_layer_outputs[j]

            output_layer_logits.append(neuron_sum)

        sample_loss = ZERO
        output_grads = []
        for i in range(19):
            diff = output_layer_logits[i] - target_vector[i]
            sample_loss += diff**2
            output_grads.append(TWO * diff)
        total_epoch_loss += sample_loss

        for i in range(19):
            bias_2[i] -= float(learning_rate * output_grads[i])
            w_row = weights_2[i]
            h_outs = hidden_layer_outputs
            for j in range(32):
                grad_w = output_grads[i] * h_outs[j]
                weights_2[i, j] -= float(learning_rate * grad_w)

        hidden_grads = [ZERO] * 32
        for j in range(32):
            error_sum = ZERO
            for i in range(19):
                error_sum += output_grads[i] * decimal.Decimal(str(weights_2[i, j]))
            hidden_grads[j] = error_sum * ReLU_derivative(
                hidden_layer_pre_activation[j]
            )

        for i in range(32):
            bias_1[i] -= float(learning_rate * hidden_grads[i])
            w_row = weights_1[i]
            for j in range(20):
                grad_w = hidden_grads[i] * input_vector[j]
                weights_1[i, j] -= float(learning_rate * grad_w)

    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:3d} | Average loss:"
            f" {total_epoch_loss / len(dataset):.8f}"
        )

print(
    f"\n Training completed in {time.time() - start_time:.2f} seconds!"
    " Checking..."
)


correct_count = 0
for input_vector, target_vector, a, b, target_output in dataset:
    h_outs = [
        ReLU(
            decimal.Decimal(str(bias_1[i]))
            + sum(
                decimal.Decimal(str(weights_1[i, j])) * input_vector[j]
                for j in range(20)
            )
        )
        for i in range(32)
    ]

    logits = [
        decimal.Decimal(str(bias_2[i]))
        + sum(
            decimal.Decimal(str(weights_2[i, j])) * h_outs[j] for j in range(32)
        )
        for i in range(19)
    ]

    predicted_sum = logits.index(max(logits))
    if predicted_sum == target_output:
        correct_count += 1
    else:
        print(
            f"Error: {a} + {b} = {target_output}, but the network said"
            f" {predicted_sum}"
        )

print(f"\nResult: Successfully predicted {correct_count} out of 100 variants!")
