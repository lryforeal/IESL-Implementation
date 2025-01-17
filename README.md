IESL is a method for finding approximate Nash equilibria in multiplayer imperfect information games. In this repository, we provide two implementations (C++ and Python) for IESL.

The directory "original" contains our C++ implementation in the poker (Kuhn poker and Leduc hold'em) and NFG (10 different normal-form games) test environments. You can directly compile and execute the programs.

The directory "open_spiel/python/algorithms" contains our Python implementation based on the OpenSpiel framework (https://github.com/google-deepmind/open_spiel). To implement IESL, we borrow the existing code of the NeuRD algorithm. To test it, you should install OpenSpiel and place the programs in the corresponding directory.

While the C++ implementation runs much faster, the Python implementation allows you to test IESL in more environments provided by OpenSpiel. If you have any questions, please contact lurunyu17@mails.ucas.ac.cn.
