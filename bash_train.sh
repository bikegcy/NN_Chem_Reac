# CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 933 --max_epochs 2000000 --learning_rate 0.001 >> record/0307_shell_test.txt
# CUDA_VISIBLE_DEVICES=1 python train.py --batch_size 933 --max_epochs 1000000 --learning_rate 0.01
# CUDA_VISIBLE_DEVICES=2 nohup python train.py --batch_size 9436 --max_epochs 2000000 --learning_rate 0.001 --input_dim '[0, 1, 4]' --output_dim '[14]' >> record/0307_3-1.txt &

# model parameters
# flags.DEFINE_string ('input_dim',      '[0, 1]', 'the input dimension')
# flags.DEFINE_string ('output_dim',     '[14]', 'the output dimension')
# flags.DEFINE_string ('layers',         '[80, 80, 80, 80, 80]', 'layers of the neural net')
# flags.DEFINE_string ('act_funcs', 'tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu', 'non-linear funcs')

# optimization1
# flags.DEFINE_float  ('learning_rate_decay', 0.5,  'learning rate decay')
# flags.DEFINE_float  ('learning_rate',       0.001,  'starting learning rate')
# flags.DEFINE_float  ('decay_when',          30.0,  'decay if loss is less than the decay_when * learning_rate')
# flags.DEFINE_integer('batch_size',          200,   'number of data to train on in parallel')
# flags.DEFINE_integer('max_epochs',          500000,   'number of full passes through the training data')

# input_names  = ['P', 'T', 'C', 'N', 'O']
#                  0    1    2    3    4
# output_names = ['H', 'He', 'C', 'N', 'O', 'H2', 'CO', 'CO2', 'CH4', 'H2O', 'N2', 'HCN', 'NH3']
#                  0    1     2    3    4    5     6      7      8      9     10    11     12

for layer in '[80,80,80,80,80]' '[100,100,100,100]' '[120,120,120,120,120]'
do
    for decay in 10 15 20 25 30
        do
            for learn_rate in 0.005 0.002 0.001 0.0008 0.0005
                do
                    CUDA_VISIBLE_DEVICES=0 nohup python train.py --batch_size 933 --max_epochs 3000 --learning_rate ${learn_rate} --layers ${layer} --decay_when ${decay} >> record/0308_${learn_rate}_${layer}_${decay}.txt &
            done
    done
done
echo all done
