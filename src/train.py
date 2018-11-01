import tensorflow as tf
import numpy as np
from DDDQNNet import DDDQNNet
from Memory import Memory
from utilities import preprocess_frame, stack_frames, stack_size, stacked_frames, update_target_graph

import client as c

def random_action(action_size, action_dict):
    index = np.random.randint(action_size)
    return action_dict[index], action_hot_encoded[index]

s = c.connect()
c.init(s)

## Discretize actions 

action_dict = {
    0: [-1, 0, 0], 
    1: [1, 0, 0], 
    2: [0, 1, 0], 
    3: [0, 0, 0.8], 
    4: [0, 0, 0]
    }

action_hot_encoded = np.identity(len(action_dict),dtype=int).tolist()

### MODEL HYPERPARAMETERS
state_size = [96,96,4]      # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels) 
action_size = len(action_dict) #game.get_available_buttons_size()              # 7 possible actions
learning_rate =  0.00025      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 5000         # Total episodes for training
max_steps = 5000              # Max possible steps in an episode
batch_size = 64             

# FIXED Q TARGETS HYPERPARAMETERS 
max_tau = 10000 #Tau is the C step where we update our target network

# EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.00005            # exponential decay rate for exploration prob

# Q LEARNING hyperparameters
gamma = 0.95               # Discounting rate

### MEMORY HYPERPARAMETERS
## If you have GPU change to 1million
pretrain_length = 10000#100000   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 10000#100000       # Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = False

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False


# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DDDQNNet(state_size, action_size, learning_rate, name="DQNetwork")

# Instantiate the target network
TargetNetwork = DDDQNNet(state_size, action_size, learning_rate, name="TargetNetwork")


# Instantiate memory
memory = Memory(memory_size)

c.reset(s)

for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        state, rw, done, action = c.step_sample(s)

        state = np.array(state)
        # First we need a state
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    
    # Get the rewards
    action, encoded = random_action(action_size, action_dict)
    next_state, reward, done = c.steps(s, action)

    next_state = np.array(next_state)

    # If we're dead
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)
        
        # Add experience to memory
        
        experience = state, encoded, reward, next_state, done
        
        memory.store(experience)
         
        c.reset(s)
        
        state, rw, done, action = c.step_sample(s)

        state = np.array(state)
        
        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:
        # Get the next state
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        
        # Add experience to memory
        experience = state, encoded, reward, next_state, done
        memory.store(experience)
        
        # Our state is now the next_state
        state = next_state

print("Finish loading memory")


# Setup TensorBoard Writer
writer = tf.summary.FileWriter("./tesorboard/")

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()

# Saver will help us to save our model
saver = tf.train.Saver()
training, restore = (True, False)

if training == True:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())
        
        if restore == True:
            print("Model Loaded")
            saver.restore(sess, "./models/car_dual.ckpt")
        # Initialize the decay rate (that will use to reduce epsilon) 
        decay_step = 0
        
        # Set tau = 0
        tau = 0

        # Init the game
        state, rw, done, action = c.step_sample(s)
        
        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = update_target_graph()
        sess.run(update_target)
        
        for episode in range(total_episodes):
            # Set step to 0
            step = 0
            
            # Initialize the rewards of the episode
            episode_rewards = []

            
            #state = env.env.state
            
            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        
            while step < max_steps:
                step += 1
                
                # Increase the C step
                tau += 1
                
                # Increase decay_step
                decay_step +=1
                
                # With ϵ select a random action atat, otherwise select a = argmaxQ(st,a)
                #action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, 
                #                                             decay_step, state)



                exp_exp_tradeoff = np.random.rand()

                # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
                explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
                
                if (explore_probability > exp_exp_tradeoff):
                    # Make a random action (exploration)
                    # Do the action
                    action, encoded = random_action(action_size, action_dict)
                    next_state, reward, done = c.steps(s, action)
                    
                else:
                    # Get action from Q-network (exploitation)
                    # Estimate the Qs values state
                    Qs = sess.run(DQNetwork.output, 
                        feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})        
                    # Take the biggest Q value (= the best action)
                    choice = np.argmax(Qs)
                    action = action_dict[int(choice)]#Qs[0]
                    encoded = action_hot_encoded[int(choice)]
                    # Do the action
                    next_state, reward, done = c.steps(s, action)
           

                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if done:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape, dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probability))

                    # Add experience to memory
                    experience = state, encoded, reward, next_state, done
                    memory.store(experience)
                    
                    c.reset(s)

                else:
                    
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    

                    # Add experience to memory
                    experience = state, encoded, reward, next_state, done
                    memory.store(experience)
                    
                    # st+1 is now our current state
                    state = next_state


                ### LEARNING PART            
                # Obtain random mini-batch from memory
                tree_idx, batch, ISWeights_mb = memory.sample(batch_size)

                states_mb = np.array([each[0][0] for each in batch], ndmin=3)
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch]) 
                next_states_mb = np.array([each[0][3] for each in batch], ndmin=3)
                dones_mb = np.array([each[0][4] for each in batch])

                target_Qs_batch = []

                
                ### DOUBLE DQN Logic
                # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                # Use TargetNetwork to calculate the Q_val of Q(s',a')
                
                # Get Q values for next_state 
                q_next_state = sess.run(DQNetwork.output, 
                feed_dict = {DQNetwork.inputs_: next_states_mb})
                
                # Calculate Qtarget for all actions that state
                q_target_next_state = sess.run(TargetNetwork.output, 
                feed_dict = {TargetNetwork.inputs_: next_states_mb})
                
                
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a') 
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    
                    # We got a'
                    action = np.argmax(q_next_state[i])

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        # Take the Qtarget for action a'
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])

                
                _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                    feed_dict={DQNetwork.inputs_: states_mb,
                                               DQNetwork.target_Q: targets_mb,
                                               DQNetwork.actions_: actions_mb,
                                              DQNetwork.ISWeights_: ISWeights_mb})
              
                
                
                # Update priority
                memory.batch_update(tree_idx, absolute_errors)
                
                
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                   DQNetwork.target_Q: targets_mb,
                                                   DQNetwork.actions_: actions_mb,
                                              DQNetwork.ISWeights_: ISWeights_mb})
                writer.add_summary(summary, episode)
                writer.flush()
                
                if tau > max_tau:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("Model updated")

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/car_dual.ckpt")
                print("Model Saved")
            if episode % 100 == 0:
                save_path = saver.save(sess, "./models/car_dual-{}.ckpt".format(episode))
                print("Model Saved steped")