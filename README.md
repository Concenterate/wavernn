
# wavernn-896

This branch consists of wavernn-896 implementation, it is an easier implementation of it and uses two rnn to generate high fidelity audio.


My implementation of RNN consists of 

	1. Sample audio into 16bit data
	2. DivMod the sample data to get 8bit questionant(coarse data) and 8bit remainder(fine data)
	3. Resample both coarse and fine data into frames of 256 units.
	3. Condition a fully dense layer using 64 bit mel spectogram of audio to get a tensor of output shape 256 (call it K)
	4. Use K to locally condition coarse rnn
	5. Use output from coarse rnn to locally condition fine rnn
	6. Resample coarse and fine rnn to get output signal
	7. Use 2 softmax, for each of coarse and fine to calculate cross entropy.
	8. Train the network using Ada-Delta optimizer.


## Tasks
- [x] Basic implementation of wavernn, can be used as vocoder easily.
- [ ] Sparse prunification.
- [ ] Trained model and some reference outputs.


## Support

Please checkout https://kdhingra307.github.io/speech_works for more details or search for issues with #rnn896. if you have any doubt or any feature request, do raise an issue.