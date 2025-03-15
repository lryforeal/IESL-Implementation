This repository contains an algorithm for finding (approximate) Nash equilibria in multiplayer games. The algorithm corresponds to the discrete implementation of IESL in [1], which is also equivalent to single-round DRDA in [2] when the base policy is uniform.

The directory "original" contains our C++ implementation in the poker (Kuhn poker and Leduc hold'em) and NFG (10 different normal-form games) test environments. You can directly compile and execute the programs.

The directory "open_spiel/python/algorithms" contains our Python implementation based on the OpenSpiel framework (https://github.com/google-deepmind/open_spiel). As we borrow the existing code of the NeuRD algorithm, please install OpenSpiel and place the programs in the corresponding directory in order to test it.

While the C++ implementation runs much faster, the Python implementation allows you to test the algorithm in more environments provided by OpenSpiel. If you have any questions, please contact lurunyu17@mails.ucas.ac.cn.

References:

[1] Runyu Lu, Yuanheng Zhu, Dongbin Zhao, Yu Liu, and You He. Last-iterate convergence to approximate Nash equilibria in multiplayer imperfect information games. IEEE Transactions on Neural Networks and Learning Systems, pp. 1–15, 2024. doi:10.1109/TNNLS.2024.3516693.

[2] Runyu Lu, Yuanheng Zhu, and Dongbin Zhao. Divergence-regularized discounted aggregation: Equilibrium finding in multiplayer partially observable stochastic games. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=KD5nJUgeW4.
