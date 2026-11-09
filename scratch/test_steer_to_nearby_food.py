from apeks.sim import Agent, World, Trail, NavMap, GRID_SIZE, build_nav, DIRECTIONS

def test_steer():
    nest = [(49, 49), (50, 49), (49, 50), (50, 50)]
    nst_nav = build_nav(nest)
    
    # Food is at (30, 47) - distance 3 from (30, 50)
    # Food trail is on (30..45, 50) - row 50 has empty trail
    a = Agent(0, 30, 50)
    a.state = "walk"
    a.role = "forager"
    
    w = World(nst_nav, [a], nest_nav=nst_nav, nest_set=set(nest), food={(30, 47): 10})
    for c in range(30, 46):
        w.food_trail.mark(c, 50, 0.0)
        
    trail = Trail()
    
    old_step = Agent.step
    def new_step(self, world, trail, now):
        if self.state == "walk" and not self.has_food:
            options = [
                (self.col + dc, self.row + dr)
                for dc, dr in DIRECTIONS
                if world.free((self.col + dc, self.row + dr))
            ]
            candidates = [t for t in options if t != self.prev_tile] or options
            nearby_active = [
                f for f, amt in world.food.items()
                if amt > 0 and abs(f[0] - self.col) + abs(f[1] - self.row) <= 6
            ]
            if nearby_active and candidates:
                target_food = min(nearby_active, key=lambda f: abs(f[0] - self.col) + abs(f[1] - self.row))
                curr_dist = abs(self.col - target_food[0]) + abs(self.row - target_food[1])
                closer = [
                    t for t in candidates
                    if abs(t[0] - target_food[0]) + abs(t[1] - target_food[1]) < curr_dist
                ]
                if closer:
                    world.move(self, min(closer, key=lambda t: abs(t[0] - target_food[0]) + abs(t[1] - target_food[1])))
                    self.role = "forager"
                    return
        return old_step(self, world, trail, now)
        
    Agent.step = new_step
    
    print("Initial pos:", a.col, a.row)
    for step in range(5):
        w.tick([a], trail, step * 0.1)
        print(f"Step {step}: pos=({a.col}, {a.row}), state={a.state}, has_food={a.has_food}")

if __name__ == "__main__":
    test_steer()
