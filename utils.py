from collections import deque
import torch
from torch.distributions import Normal



class QueueAsStack:
    def __init__(self, max_size=10):
        self.queue = deque(maxlen=max_size)

    def push(self, item):
        while len(self.queue) >= self.queue.maxlen:
            self.queue.popleft()  # If the queue is full, remove the oldest element
        self.queue.append(item)

    def pop(self):
        if len(self.queue) > 0:
            return self.queue.pop()
        else:
            print("Stack is empty.")

    def mean(self):
        return sum(self.queue) / len(self.queue)


def select_action(encoder, actor, obs, device):
    with torch.no_grad():
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        state = encoder(obs)
        mu, _ = actor(state)
        action = mu.squeeze(0).detach().cpu().numpy()
    return action


def normalization(a):
    return (a - a.mean())/(a.std() + 1e-10)


def write_array_to_txt(arr, file_path):
    # 将NumPy数组写入txt文件
    with open(file_path, 'w') as f:
        for value in arr:
            f.write(f"{value}\n")
    print(f"NumPy array successfully written to file {file_path}")



class SuccessRateWriter:
    def __init__(self, n, directory="."):
        """
        Initialize an instance of the SuccessRateWrite class and create a success_rate.txt file in the specified directory.
        : paramn: The number of rows in the matrix
        : paramdirectory: The directory where the file is created, defaults to the current directory (".")
        """
        assert n > 0, "n must be greater than 0"
        self.n = n
        self.matrix = [[0]*2 for _ in range(n)] 
        self.file_path = f"{directory}/success_rate.txt"
        
        # 创建文件
        with open(self.file_path, "w") as file:
            file.write("")  

        self.data_len = 0

    def write_data(self, data_value, step):
        self.matrix[self.data_len][0] = step
        self.matrix[self.data_len][1] = data_value
        self.data_len += 1
        if self.data_len == self.n:
            return True
        return False

    def write_matrix_to_file(self):
        
        with open(self.file_path, "a") as file:
            for row in range(self.data_len):
                file.write(f"{self.matrix[row][0]},{self.matrix[row][1]}")
                file.write("\n")

    def clear_matrix(self):
        
        self.matrix = [[0]*2 for _ in range(self.n)]
        self.data_len = 0
