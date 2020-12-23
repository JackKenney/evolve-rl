package internal

// EpisodeTracker struct tracks (s,a,r) tuples for episodic agent updates
type EpisodeTracker struct {
	t       int           // Timestep in current episode
	N       int           // How many episodes before update?
	epCount int           // Current episode
	states  [][][]float64 // States history
	actions [][]int       // Action history
	rewards [][]float64   // Rewards history
}

// NewEpisodeTracker returns new episodes
func NewEpisodeTracker(N int) *EpisodeTracker {
	ep := EpisodeTracker{}
	ep.Wipe()
	ep.N = N

	return &ep
}

// Update tracker with new tuple.
func (ep *EpisodeTracker) Update(s []float64, a int, r float64, sPrime []float64) {
	ep.states[ep.epCount] = append(ep.states[ep.epCount], s)
	ep.actions[ep.epCount] = append(ep.actions[ep.epCount], a)
	ep.rewards[ep.epCount] = append(ep.rewards[ep.epCount], r)
	ep.t++
}

// LastUpdate makes final update. Returns true if Nth episode just finished and agent should be ready for update.
func (ep *EpisodeTracker) LastUpdate(s []float64, a int, r float64) bool {
	ep.states[ep.epCount] = append(ep.states[ep.epCount], s)
	ep.actions[ep.epCount] = append(ep.actions[ep.epCount], a)
	ep.rewards[ep.epCount] = append(ep.rewards[ep.epCount], r)

	// Increment episode counter
	ep.epCount++
	ep.t = 0

	// If ready to update, update and wipe the states, actions, and rewards
	if ep.epCount == ep.N {
		return true
	}
	return false
}

// Wipe the contents of the tracker.
func (ep *EpisodeTracker) Wipe() {
	ep.t = 0
	ep.epCount = 0
	ep.states = make([][][]float64, ep.N)
	ep.actions = make([][]int, ep.N)
	ep.rewards = make([][]float64, ep.N)
}
