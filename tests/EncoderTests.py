import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/PositionalEncodings/")))

import torch
import numpy as np
import unittest
from Encoders import PositionalEncoding1D, PositionalEncoding2D


class TestPositionalEncoding(unittest.TestCase):

    def test_output_shape_1D(self):
        batch_size, seq_length, enc_dim = 5, 10, 1
        x = torch.randn(batch_size, seq_length, enc_dim)
        pos_enc = PositionalEncoding1D(seq_length, enc_dim)
        out = pos_enc(x)
        self.assertEqual(out.shape, x.shape)

    def test_output_shape_2D(self):
        height, width, channels = 32, 32, 3
        batch_size = 5
        x = torch.randn(batch_size, channels, height, width)
        pos_enc = PositionalEncoding2D(height, width, channels)
        out = pos_enc(x)
        self.assertEqual(out.shape, x.shape)

    def test_correct_encoding_1D(self):
        batch_size, seq_length, enc_dim = 5, 10, 1
        x = torch.randn(batch_size, seq_length, enc_dim)
        pos_enc = PositionalEncoding1D(seq_length, enc_dim)
        out = pos_enc(x)
        expected_out = x + pos_enc.pos_encoding.to(x.device)
        self.assertTrue(torch.allclose(out, expected_out))

    def test_correct_encoding_2D(self):
        height, width, channels = 32, 32, 3
        x = torch.randn(1, channels, height, width)
        pos_enc = PositionalEncoding2D(height, width, channels)
        out = pos_enc(x)
        expected_out = (x.permute(0, 2, 3, 1) + pos_enc.pos_encoding.to(x.device)).permute(0, 3, 1, 2)
        self.assertTrue(torch.allclose(out, expected_out))

    def test_device_agnostic_1D(self):
        batch_size, seq_length, enc_dim = 5, 10, 1
        x = torch.randn(batch_size, seq_length, enc_dim)
        pos_enc = PositionalEncoding1D(seq_length, enc_dim)
        out_cpu = pos_enc(x)

        if torch.cuda.is_available():
            x_gpu = x.to("cuda")
            pos_enc_gpu = PositionalEncoding1D(seq_length, enc_dim).to("cuda")
            out_gpu = pos_enc_gpu(x_gpu)
            self.assertEqual(out_cpu.shape, out_gpu.shape)

    def test_device_agnostic_2D(self):
        height, width, channels = 32, 32, 3
        x = torch.randn(1, channels, height, width)
        pos_enc = PositionalEncoding2D(height, width, channels)
        out_cpu = pos_enc(x)

        if torch.cuda.is_available():
            x_gpu = x.to("cuda")
            pos_enc_gpu = PositionalEncoding2D(height, width, channels).to("cuda")
            out_gpu = pos_enc_gpu(x_gpu)
            self.assertEqual(out_cpu.shape, out_gpu.shape)

if __name__ == '__main__':
    unittest.main()
