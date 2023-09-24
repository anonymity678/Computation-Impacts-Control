import numpy as np
import pickle
import jax, jax.numpy as jnp, jax.experimental.ode, optax
from tensorboardX import SummaryWriter
from tqdm import tqdm

horizon_tau = 2
tau = None
horizon = None
num_steps = 30000
# rng = np.random.default_rng(seed=3)

# init_state = rng.uniform(low=-0.2, high=0.2, size=(4,))
# init_state = jnp.array(init_state)
init_state = jnp.array([-0.1, -0.1,  0.1,  0.1])
target_state = jnp.array([0., 0., 0., 0.])

Q = jnp.array([0.1, 1.0, 100.0, 5.0])
R = jnp.array([0.1])

def save_policy(policy, filename):
    with open(filename, 'wb') as f:
        pickle.dump(policy, f)

def dynamics(t: float, state: jnp.ndarray, action: jnp.ndarray):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5
    polemass_length = masspole * length
    force_mag = 1.0
    force = force_mag * action.squeeze()
    x, x_dot, theta, theta_dot = state

    # Note: everything below this is same as gym's cartpole step fun.
    costheta = 1
    sintheta = theta
    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass
    return jnp.array([x_dot, xacc, theta_dot, thetaacc])

def create_policy(key):
    input, hidden, output = 5, 8, 1
    init = jax.nn.initializers.glorot_normal()
    return {
        'w1': init(key, (input, hidden)),
        'b1': jnp.zeros((hidden,)),
        'w2': init(key, (hidden, hidden)),
        'b2': jnp.zeros((hidden,)),
        'w3': init(key, (hidden, output)),
        'b3': jnp.zeros((output,))
    }

def forward_policy(policy, x):
    x = x @ policy['w1'] + policy['b1']
    x = jax.nn.gelu(x)
    x = x @ policy['w2'] + policy['b2']
    x = jax.nn.gelu(x)
    x = x @ policy['w3'] + policy['b3']
    return x

def get_action(t: float, state: jnp.ndarray, policy: dict):
    x = jnp.append(state, t)
    x = jnp.expand_dims(x, axis=0)
    x = forward_policy(policy, x)
    x = x.squeeze(0)
    return x

def get_loss(state, action):
    return jnp.dot(action, R * action) + jnp.dot(state, Q * state)

def rollout_euler(policy):
    def autonomous_euler(t: float, state: jnp.ndarray):
        action = get_action(t, state, policy)
        return t + tau, state + tau * dynamics(t, state, action), action

    t, state = 0, init_state
    init_action = get_action(t, init_state, policy)
    loss_sum = get_loss(init_state, init_action)
    loss_sum *= (0.5*tau)
    for i in range(1, horizon+1):
        t, state, action = autonomous_euler(t, state)
        loss = get_loss(state, action)
        if i==horizon:
            loss *= (0.5*tau)
        else:
            loss *= tau
        loss_sum += loss
    return loss_sum

def rollout_odeint(policy):
    def autonomous_ode(state_aug: jnp.ndarray, t: float):
        state = state_aug[:-1]
        action = get_action(t, state, policy)
        return jnp.append(dynamics(t, state, action), get_loss(state, action))

    init_state_aug = jnp.append(init_state, 0.0)
    time_range = jnp.array([0, horizon_tau], dtype=float)
    states = jax.experimental.ode.odeint(autonomous_ode, init_state_aug, time_range)
    loss = states[-1, -1]
    return loss

@jax.jit
def rollout_evaluation(policy):
    def autonomous_ode(state_aug: jnp.ndarray, t: float):
        state = state_aug[:-1]
        action = get_action(t, state, policy)
        return jnp.append(dynamics(t, state, action), get_loss(state, action))

    time_range_eval = jnp.array([0, horizon_tau], dtype=float)

    init_y = jnp.append(init_state, 0.0)
    states = jax.experimental.ode.odeint(autonomous_ode, init_y, time_range_eval)
    loss_eval = states[-1, -1]
    return loss_eval

def main_autograd():
    my_writer = SummaryWriter(flush_secs=10)
    policy = create_policy(jax.random.PRNGKey(0))
    optimizer = optax.adam(3e-5)
    opt_state = optimizer.init(policy)

    min_loss = float('inf')
    best_policy = None
    global horizon
    horizon = int(horizon_tau / tau)

    @jax.jit
    def learn_autograd(policy, opt_state):
        loss, grads = jax.value_and_grad(rollout_euler, argnums=0)(policy)
        updates, opt_state = optimizer.update(grads, opt_state, policy)
        policy = optax.apply_updates(policy, updates)
        return policy, opt_state, loss

    progress = tqdm(range(num_steps))
    for step in progress:
        policy, opt_state, loss = learn_autograd(policy, opt_state)
        loss_eval = rollout_evaluation(policy)

        if loss_eval < min_loss:
            min_loss = loss_eval
            best_policy = policy

        progress.set_postfix({'loss': loss_eval})
        my_writer.add_scalar(f"loss_tau_{tau}", loss_eval, step)

    # save_policy(best_policy, f'autograd_policy_tau_{tau}.pkl')

def main_odeint():
    my_writer = SummaryWriter(flush_secs=10)
    policy = create_policy(jax.random.PRNGKey(0))
    optimizer = optax.adam(3e-5)
    opt_state = optimizer.init(policy)
    
    min_loss = float('inf')
    best_policy = None

    @jax.jit
    def learn_odeint(policy, opt_state):
        loss, grads = jax.value_and_grad(rollout_odeint, argnums=0)(policy)
        updates, opt_state = optimizer.update(grads, opt_state, policy)
        policy = optax.apply_updates(policy, updates)
        return policy, opt_state, loss

    progress = tqdm(range(num_steps))
    for step in progress:
        policy, opt_state, loss = learn_odeint(policy, opt_state)
        loss_eval = rollout_evaluation(policy)

        if loss_eval < min_loss:
            min_loss = loss_eval
            best_policy = policy
            
        progress.set_postfix({'loss':loss_eval})
        my_writer.add_scalar("loss", loss_eval, step)

    # save_policy(best_policy, 'odeint_policy.pkl')

if __name__ == '__main__':
    tau_values = [0.01, 0.02, 0.04]  # Replace with your list of tau values
    for item in tau_values:
        tau = item
        main_autograd()
    main_odeint()