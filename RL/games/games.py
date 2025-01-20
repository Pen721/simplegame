from games.ResourceGame import ResourceGame
import numpy as np

class ActionOneBetterThanActionZero(ResourceGame):
    """
    Action 1 is U(0, 999) when resources > 0, -1 when resources <= 0. Therefore should always gamble when resources > 0.
    """
    def __init__(self, max_steps=100):
        super().__init__(max_steps)

    def update_states(self, action):
        super().update_states(action)
        old_resources = self.resources

        if action == 1:  # Spend
            if self.resources > 0:
                self.resources += np.random.randint(10, 100)
                self.resources -= 1
            else:
                self.resources -= 1
        else:  # Save
            self.resources += 1

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        # Calculate the change in resources
        reward = self.resources - old_resources
        return reward
    
class NotSoSimpleResourceGame(ResourceGame):
    def __init__(self, max_steps=100):
        super().__init__(max_steps)

    def update_states(self, action):
        old_resources = self.resources

        if action == 1:  # Spend
            if self.resources > 0:
                # reward signal only slightly better than save
                self.resources += np.random.randint(0, 6)
                self.resources -= 1
            else:
                self.resources -= 1
        else:  # Save
            self.resources += 1

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        # Calculate the change in resources
        reward = self.resources - old_resources
        return reward
    
class AlternatingRewardResourceGame(ResourceGame):
    def __init__(self, max_steps=100):
        super().__init__(max_steps)

    def update_states(self, action):
        old_resources = self.resources

        smallReward = (self.step_count // 10) % 2 == 0
        if action == 1:  # Spend
            if self.resources > 0:
                # reward signal only slightly better than save
                if smallReward:
                    self.resources += np.random.randint(0, 4) #slightly worse than saving
                else:
                    self.resources += np.random.randint(0, 10) # way better than saving
                self.resources -= 1
            else:
                self.resources -= 1
        else:  # Save
            self.resources += 1

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        # Calculate the change in resources
        reward = self.resources - old_resources
        return reward
    
class SpendingGoodInBeginning(ResourceGame):
    def __init__(self, max_steps=100):
        super().__init__(max_steps)

    def update_states(self, action):
        old_resources = self.resources

        smallReward = self.step_count > 50
        if action == 1:  # Spend
            if self.resources > 0:
                # reward signal only slightly better than save
                if smallReward:
                    self.resources += np.random.randint(0, 4) #slightly worse than saving
                else:
                    self.resources += np.random.randint(0, 10) # way better than saving
                self.resources -= 1
            else:
                self.resources -= 1
        else:  # Save
            self.resources += 1

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        # Calculate the change in resources
        reward = self.resources - old_resources
        return reward
    
class SpendingGoodInEnd(ResourceGame):
    def __init__(self, max_steps=100):
        super().__init__(max_steps)

    def update_states(self, action):
        old_resources = self.resources

        smallReward = self.step_count.step < 50
        if action == 1:  # Spend
            if self.resources > 0:
                # reward signal only slightly better than save
                if smallReward:
                    self.resources += np.random.randint(0, 4) #slightly worse than saving
                else:
                    self.resources += np.random.randint(0, 10) # way better than saving
                self.resources -= 1
            else:
                self.resources -= 1
        else:  # Save
            self.resources += 1

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        # Calculate the change in resources
        reward = self.resources - old_resources
        return reward

class OnlySpend10TimesResourceGame(ResourceGame):
    # Model will only get rewarded for the first ten spends, then after that it deterioates
    def __init__(self, max_steps=100):
        super().__init__(max_steps)

    def update_states(self, action):
        spend_count = self.actions.count(1)

        old_resources = self.resources

        smallReward = spend_count > 10

        if action == 1:  # Spend
            if self.resources > 0:
                # reward signal only slightly better than save
                if smallReward:
                    self.resources += np.random.randint(0, 4) #slightly worse than saving
                else:
                    self.resources += np.random.randint(0, 10) # way better than saving
                self.resources -= 1
            else:
                self.resources -= 1
        else:  # Save
            self.resources += 1

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        # Calculate the change in resources
        reward = self.resources - old_resources
        return reward
    
class NonMarkovGame(ResourceGame):
    # Model will only get rewarded for the first ten spends, then after that it deterioates
    def __init__(self, max_steps=100):
        super().__init__(max_steps)

    def update_states(self, action):
        spend_count = self.actions.count(1)

        old_resources = self.resources

        smallReward = spend_count > 10
        
        if action == 1:  # Spend
            if self.resources > 0:
                # reward signal only slightly better than save
                if smallReward:
                    self.resources += np.random.randint(0, 4) #slightly worse than saving
                else:
                    self.resources += np.random.randint(0, self.step_count + 1) # way better than saving in later game
                self.resources -= 1
            else:
                self.resources -= 1
        else:  # Save
            self.resources += 1

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        # Calculate the change in resources
        reward = self.resources - old_resources
        return reward