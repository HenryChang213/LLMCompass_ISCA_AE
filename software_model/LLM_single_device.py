from software_model.operators import (
    Operator,
    Tensor,
    Matmul,
    Reshape,
    Concat,
    Transpose,
    Softmax,
    GeLU,
)

from utils import size

# TODO: embedding layer


class TransformerBlockInitComputation(Operator):
    def __init__(self, b, l, d, h):
        super().__init__(0, 0, 0, 0)
        # parameters
        Wq = Tensor([d, d])
        Wk = Tensor([d, d])
        Wv = Tensor([d, d])
        Wo = Tensor([d, d])
        W1 = Tensor([d, 4 * d])
        W2 = Tensor([4 * d, d])
        self.Parameters = [Wq, Wk, Wv, Wo, W1, W2]
        # operators
        Q_proj = Matmul()
        K_proj = Matmul()
        V_proj = Matmul()
        Q_reshape = Reshape()
        K_reshape = Reshape()
        V_reshape = Reshape()
        Q_transpose = Transpose()
        K_transpose = Transpose()
        V_transpose = Transpose()
        Q_mul_K = Matmul()
        A_softmax = Softmax()
        A_mul_V = Matmul()
        H_transpose = Transpose()
        H_reshape = Reshape()
        H_matmul0 = Matmul()
        H_matmul1 = Matmul()
        H_gelu = GeLU()
        H_matmul2 = Matmul()
        # forward
        input = Tensor([b, l, d])
        peak_memory_usage = size(input)  # excluding parameters
        Q = Q_proj(input, Wq)
        K = K_proj(input, Wk)
        V = V_proj(input, Wv)
        peak_memory_usage = max(
            peak_memory_usage, size(Q) + size(K) + size(V) + size(input)
        )
        Q = Q_reshape(Q, [b, l, h, d / h])
        K = K_reshape(K, [b, l, h, d / h])
        V = V_reshape(V, [b, l, h, d / h])
        peak_memory_usage = max(peak_memory_usage, size(Q) + size(K) + size(V))
        Q_T = Q_transpose(Q, [b, h, l, d / h])
        K_T = K_transpose(K, [b, h, d / h, l])
        V_T = V_transpose(V, [b, h, l, d / h])
        peak_memory_usage = max(
            peak_memory_usage, size(Q) * 2 + size(K) * 2 + size(V) * 2
        )
        A = Q_mul_K(Q_T, K_T)
        peak_memory_usage = max(
            peak_memory_usage, size(Q_T) + size(K_T) + size(A) + size(V_T)
        )
        A = A_softmax(A)
        peak_memory_usage = max(peak_memory_usage, size(A) + size(V_T))
        H = A_mul_V(A, V_T)
        peak_memory_usage = max(peak_memory_usage, size(A) + size(V_T) + size(H))
        H = H_transpose(H, [b, l, h, d / h])
        peak_memory_usage = max(peak_memory_usage, size(H) * 2)
        H = H_reshape(H, [b, l, d])
        peak_memory_usage = max(peak_memory_usage, size(H))
        H = H_matmul0(H, Wo)
        peak_memory_usage = max(peak_memory_usage, size(H) * 2)
        H1 = H_matmul1(H, W1)
        peak_memory_usage = max(peak_memory_usage, size(H) + size(H1))
        H1 = H_gelu(H1)
        peak_memory_usage = max(peak_memory_usage, size(H1))
        H2 = H_matmul2(H1, W2)

        peak_memory_usage = max(peak_memory_usage, size(H1) + size(H2))
        self.peak_memory_usage = peak_memory_usage
        self.parameter_memory_usage = 0
        for i in self.Parameters:
            self.peak_memory_usage += size(i)
            self.parameter_memory_usage += size(i)
        self.Operators = [
            Q_proj,
            K_proj,
            V_proj,
            Q_reshape,
            K_reshape,
            V_reshape,
            Q_transpose,
            K_transpose,
            V_transpose,
            Q_mul_K,
            A_softmax,
            A_mul_V,
            H_transpose,
            H_reshape,
            H_matmul0,
            H_matmul1,
            H_gelu,
            H_matmul2,
        ]
        for i in self.Operators:
            self.flop_count += i.flop_count
            self.load_count += i.load_count
            self.store_count += i.store_count
            self.io_count += i.io_count


class LLMInitComputation(Operator):
    def __init__(self, d_model, n_heads, n_layers, batch_size, seq_length):
        super().__init__(0, 0, 0, 0)
        self.block = TransformerBlockInitComputation(
            batch_size, seq_length, d_model, n_heads
        )
        self.flop_count = self.block.flop_count * n_layers
        self.load_count = self.block.load_count * n_layers
        self.store_count = self.block.store_count * n_layers
        self.parameter_memory_usage = self.block.parameter_memory_usage * n_layers
        self.peak_memory_usage = self.parameter_memory_usage + (
            self.block.peak_memory_usage - self.block.parameter_memory_usage
        )
        self.io_count = self.load_count + self.store_count

    def print_info(self):
        print(
            f"{'flop_count(T)':^13}, {'load_count':^12}, {'store_count':^12}, {'parameter_memory_usage':^20}, {'peak_memory_usage':^20}"
        )
        print(
            f"{self.flop_count / 1024**4:^13.1f}, {self.load_count / 1024**3:^12.1f}, {self.store_count / 1024**3:^12.1f}, {self.parameter_memory_usage / 1024**3:^20.1f}, {self.peak_memory_usage / 1024**3:^20.1f}"
        )


class TransformerBlockAutoRegression(Operator):
    def __init__(self, b, l, d, h):
        super().__init__(0, 0, 0, 0)
        # parameters
        Wq = Tensor([d, d])
        Wk = Tensor([d, d])
        Wv = Tensor([d, d])
        Wo = Tensor([d, d])
        W1 = Tensor([d, 4 * d])
        W2 = Tensor([4 * d, d])
        self.Parameters = [Wq, Wk, Wv, Wo, W1, W2]
        # operators
        q_proj = Matmul()
        k_proj = Matmul()
        v_proj = Matmul()
        q_reshape = Reshape()
        k_reshape = Reshape()
        v_reshape = Reshape()
        K_concat = Concat()
        V_concat = Concat()
        q_mul_K = Matmul()
        a_softmax = Softmax()
        a_mul_V = Matmul()
        h_reshape = Reshape()
        h_matmul0 = Matmul()
        h_matmul1 = Matmul()
        h_gelu = GeLU()
        h_matmul2 = Matmul()
        # input
        input = Tensor([b, 1, d])
        K_cache = Tensor([b, h, d / h, l])
        V_cache = Tensor([b, h, l, d / h])
        peak_memory_usage = size(input)  # excluding parameters
        self.KV_cache_size = size(K_cache) + size(V_cache)
        # forward
        q = q_proj(input, Wq)
        k = k_proj(input, Wk)
        v = v_proj(input, Wv)
        q = q_reshape(q, [b, h, 1, d / h])
        k = k_reshape(k, [b, h, d / h, 1])
        v = v_reshape(v, [b, h, 1, d / h])
        peak_memory_usage = max(
            peak_memory_usage, size(q) + size(k) + size(v) + size(input)
        )
        K = K_concat(K_cache, k, 3)
        V = V_concat(V_cache, v, 2)
        peak_memory_usage = max(peak_memory_usage, size(K) * 2 + size(V) * 2 + size(q))
        a = q_mul_K(q, K)
        peak_memory_usage = max(
            peak_memory_usage, size(q) + size(K) + size(a) + size(V)
        )
        a = a_softmax(a)
        peak_memory_usage = max(peak_memory_usage, size(a) + size(V))
        h = a_mul_V(a, V)
        peak_memory_usage = max(peak_memory_usage, size(a) + size(V) + size(h))
        h = h_reshape(h, [b, 1, d])
        peak_memory_usage = max(peak_memory_usage, size(h))
        h = h_matmul0(h, Wo)
        peak_memory_usage = max(peak_memory_usage, size(h) * 2)
        h1 = h_matmul1(h, W1)
        peak_memory_usage = max(peak_memory_usage, size(h) + size(h1))
        h1 = h_gelu(h1)
        peak_memory_usage = max(peak_memory_usage, size(h1))
        h2 = h_matmul2(h1, W2)
        peak_memory_usage = max(peak_memory_usage, size(h1) + size(h2))
        self.peak_memory_usage = peak_memory_usage + self.KV_cache_size
        self.parameter_memory_usage = 0
        for i in self.Parameters:
            self.peak_memory_usage += size(i)
            self.parameter_memory_usage += size(i)
        self.Operators = [
            q_proj,
            k_proj,
            v_proj,
            q_reshape,
            k_reshape,
            v_reshape,
            K_concat,
            V_concat,
            q_mul_K,
            a_softmax,
            a_mul_V,
            h_reshape,
            h_matmul0,
            h_matmul1,
            h_gelu,
            h_matmul2,
        ]
        for i in self.Operators:
            self.flop_count += i.flop_count
            self.load_count += i.load_count
            self.store_count += i.store_count


class LLMAutoRegression(Operator):
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        batch_size,
        input_seq_length,
        output_seq_length,
    ):
        super().__init__(0, 0, 0, 0)
        self.block_first_token = TransformerBlockAutoRegression(
            batch_size, input_seq_length + 1, d_model, n_heads
        )
        self.block_last_token = TransformerBlockAutoRegression(
            batch_size, input_seq_length + output_seq_length - 1, d_model, n_heads
        )
        self.flop_count = (
            (self.block_first_token.flop_count + self.block_last_token.flop_count)
            / 2
            * n_layers
        )
        self.load_count = (
            (self.block_first_token.load_count + self.block_last_token.flop_count)
            / 2
            * n_layers
        )
        self.store_count = (
            (self.block_first_token.store_count + self.block_last_token.store_count)
            / 2
            * n_layers
        )
        self.parameter_memory_usage = (
            self.block_first_token.parameter_memory_usage * n_layers
        )
        self.peak_memory_usage = self.parameter_memory_usage + (
            self.block_last_token.peak_memory_usage
            - self.block_last_token.parameter_memory_usage
        )
        self.io_count = self.load_count + self.store_count

    def print_info(self):
        print(
            f"{'flop_count':^13}, {'load_count':^12}, {'store_count':^12}, {'parameter_memory_usage':^20}, {'peak_memory_usage':^20}"
        )
        print(
            f"{self.flop_count / 1024**3:^13.1f}, {self.load_count / 1024**3:^12.1f}, {self.store_count / 1024**3:^12.1f}, {self.parameter_memory_usage / 1024**3:^20.1f}, {self.peak_memory_usage / 1024**3:^20.1f}"
        )


class LLM(Operator):
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        batch_size,
        input_seq_length,
        output_seq_length,
    ):
        super().__init__(0, 0, 0, 0)
        self.initComputation = LLMInitComputation(
            d_model, n_heads, n_layers, batch_size, input_seq_length
        )
        self.autoRegression = LLMAutoRegression(
            d_model, n_heads, n_layers, batch_size, input_seq_length, output_seq_length
        )

    def print_info(self):
        print(f"{'initComputation':*^70}")
        self.initComputation.print_info()
        print(f"{'autoRegression':*^70}")
        self.autoRegression.print_info()
