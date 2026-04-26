# Task 11 — Final Explanation Report
## Student Performance Evaluator using ANN

---

### 1. What is ANN in Your Own Words?

An **Artificial Neural Network (ANN)** is a computer system loosely inspired by how the human brain works. It consists of layers of small units called **neurons**, connected to each other. Each connection has a **weight** — a number that controls how strongly one neuron influences another.

When we give the network some input (e.g., a student's attendance, quiz marks, etc.), the data flows through the network layer by layer. At each layer, every neuron does a simple calculation: it multiplies each incoming value by its weight, adds a bias, and passes the result through an **activation function** (like ReLU). After passing through all the layers, the final output gives us a prediction.

The "learning" happens during **training**: the network looks at many examples, checks how wrong its predictions are (the *loss*), and nudges the weights slightly in the direction that reduces the error — this process is called **backpropagation with gradient descent**. After thousands of such adjustments, the weights settle on values that allow the network to make accurate predictions.

---

### 2. What Function Did Your Model Learn?

Our model learned a function of the form:

```
f(attendance, assignment, quiz, mid, study_hours) → {0, 1}
```

Where:
- **0** means the student is predicted to **Fail**
- **1** means the student is predicted to **Pass**

Internally, the network learned:
- How much **attendance** matters relative to quiz scores
- The combined effect of **study_hours** and **mid-term marks**
- Non-linear thresholds — e.g., "a student who studies 10+ hours *and* has >75% attendance is very likely to pass"

This is not a simple formula we wrote — the network discovered these relationships automatically from 600 student records.

---

### 3. How Does Your System Evaluate a New Student?

When a new student's data is entered (via the UI or CLI):

1. **Input** — The five feature values are collected.
2. **Scaling** — The values are transformed using the **same StandardScaler** that was fitted on training data (mean and std from training set). This ensures the model receives inputs in the same range it was trained on.
3. **Forward Pass** — The scaled values are fed through the ANN:
   - Input → Hidden Layer 1 (32 neurons, ReLU) → Hidden Layer 2 (16 neurons, ReLU) → Output
4. **Prediction** — The output layer produces a probability for each class (Fail / Pass). The class with higher probability is chosen as the final prediction.
5. **Result** — The system returns the label ("Pass" or "Fail"), the confidence percentage, and a performance band (Low / Medium / High).

---

### 4. Why Is Scaling Important?

Scaling (standardisation) is critical in ANNs for several reasons:

| Without Scaling | With Scaling |
|---|---|
| Features with large ranges (e.g. attendance 0–100) dominate weight updates | All features contribute equally |
| Gradient descent takes very long to converge | Faster and more stable convergence |
| Model may get stuck in poor local minima | Better generalisation |
| Weight initialisation assumptions are violated | Works well with default weight init |

Our dataset has `attendance` ranging 0–100 and `study_hours` ranging 0–20. Without scaling, the network would pay disproportionate attention to attendance simply because its numbers are larger, even if study_hours is equally important.

**StandardScaler** subtracts the mean and divides by the standard deviation:
```
z = (x - mean) / std
```
This brings every feature to approximately mean=0, std=1.

---

### 5. What Are the Limitations of Your Model?

| Limitation | Explanation |
|---|---|
| **Small dataset** | Only 600 records — a larger dataset would improve generalisation |
| **Binary output only** | Predicts only Pass/Fail; cannot predict exact marks or GPA |
| **No temporal data** | Does not consider improvement over time (trend) |
| **Feature limitations** | Missing important factors like socioeconomic background, previous grades, mental health |
| **Black box** | Hard to explain *why* a specific student is predicted to fail |
| **Imbalanced risk** | False Negatives (predicting Pass when student will Fail) can be harmful if not flagged |
| **Overfitting risk** | With a small dataset and complex networks, the model might memorise rather than generalise |

---

### Model Summary

| Property | Value |
|---|---|
| Algorithm | MLPClassifier (Multi-Layer Perceptron) |
| Hidden Layers | 2 (32 neurons, 16 neurons) |
| Activation | ReLU (hidden), Softmax (output) |
| Optimiser | Adam |
| Training Split | 80% train / 20% test |
| Scaler | StandardScaler |
| Input Features | 5 |
| Output Classes | 2 (Pass / Fail) |
