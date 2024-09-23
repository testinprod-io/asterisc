package fast

import (
	"encoding/csv"
	"math/bits"
	"os"
	"strconv"
	"sync"
)

// RadixNode is an interface defining the operations for a node in a radix trie.
type RadixNode interface {
	// InvalidateNode invalidates the hash cache along the path to the specified address.
	InvalidateNode(addr uint64)
	// GenerateProof generates the Merkle proof for the given address.
	GenerateProof(addr uint64) [][32]byte
	// MerkleizeNode computes the Merkle root hash for the node at the given generalized index.
	MerkleizeNode(addr, gindex uint64) [32]byte
}

const SmallRadixSize = 4

// SmallRadixNode is a radix trie node with a branching factor of 4 bits.
type SmallRadixNode[C RadixNode] struct {
	Stats       *Stats
	Children    [1 << SmallRadixSize]*C       // Array of child nodes, indexed by 4-bit keys.
	Hashes      [1 << SmallRadixSize][32]byte // Cached hashes for each child node.
	ChildExists uint16                        // Bitmask indicating which children exist (1 bit per child).
	HashValid   uint16                        // Bitmask indicating which hashes are valid (1 bit per child).
	Depth       uint64                        // The depth of this node in the trie (number of bits from the root).
}

const LargeRadixSize = 8

// LargeRadixNode is a radix trie node with a branching factor of 8 bits.
type LargeRadixNode[C RadixNode] struct {
	Stats       *Stats
	Children    [1 << LargeRadixSize]*C // Array of child nodes, indexed by 8-bit keys.
	Hashes      [1 << LargeRadixSize][32]byte
	ChildExists [(1 << LargeRadixSize) / 64]uint64
	HashValid   [(1 << LargeRadixSize) / 64]uint64
	Depth       uint64
}

// Define a sequence of radix trie node types (L1 to L11) representing different levels in the trie.
// Each level corresponds to a node type, where L1 is the root node and L11 is the leaf level pointing to Memory.
// The cumulative bit-lengths of the addresses represented by the nodes from L1 to L11 add up to 52 bits.

type L1 = SmallRadixNode[L2]
type L2 = *SmallRadixNode[L3]
type L3 = *SmallRadixNode[L4]
type L4 = *SmallRadixNode[L5]
type L5 = *SmallRadixNode[L6]
type L6 = *SmallRadixNode[L7]
type L7 = *SmallRadixNode[L8]
type L8 = *LargeRadixNode[L9]
type L9 = *LargeRadixNode[L10]
type L10 = *LargeRadixNode[L11]
type L11 = *Memory

// InvalidateNode invalidates the hash cache along the path to the specified address.
// It marks the necessary child hashes as invalid, forcing them to be recomputed when needed.
func (n *SmallRadixNode[C]) InvalidateNode(addr uint64) {
	childIdx := addressToRadixPath(addr, n.Depth, SmallRadixSize) // Get the 4-bit child index at the current depth.

	branchIdx := (childIdx + 1<<SmallRadixSize) / 2 // Compute the index for the hash tree traversal.

	// Traverse up the hash tree, invalidating hashes along the way.
	for index := branchIdx; index > 0; index >>= 1 {
		hashBit := index & 15          // Get the relevant bit position (0-15).
		n.ChildExists |= 1 << hashBit  // Mark the child as existing.
		n.HashValid &= ^(1 << hashBit) // Invalidate the hash at this position.
	}
}

func (n *LargeRadixNode[C]) InvalidateNode(addr uint64) {
	childIdx := addressToRadixPath(addr, n.Depth, LargeRadixSize)

	branchIdx := (childIdx + 1<<LargeRadixSize) / 2

	for index := branchIdx; index > 0; index >>= 1 {
		hashIndex := index >> 6
		hashBit := index & 63
		n.ChildExists[hashIndex] |= 1 << hashBit
		n.HashValid[hashIndex] &= ^(1 << hashBit)
	}
}

func (m *Memory) InvalidateNode(addr uint64) {
	if p, ok := m.pageLookup(addr >> PageAddrSize); ok {
		p.Invalidate(addr & PageAddrMask)
	}
}

// GenerateProof generates the Merkle proof for the given address.
// It collects the necessary sibling hashes along the path to reconstruct the Merkle proof.
func (n *SmallRadixNode[C]) GenerateProof(addr uint64) [][32]byte {
	var proofs [][32]byte
	path := addressToRadixPath(addr, n.Depth, SmallRadixSize)

	if n.Children[path] == nil {
		// When no child exists at this path, the rest of the proofs are zero hashes.
		proofs = zeroHashRange(0, 60-n.Depth-SmallRadixSize)
	} else {
		// Recursively generate proofs from the child node.
		proofs = (*n.Children[path]).GenerateProof(addr)
	}

	// Collect sibling hashes along the path for the proof.
	for idx := path + 1<<SmallRadixSize; idx > 1; idx >>= 1 {
		sibling := idx ^ 1 // Get the sibling index.
		proofs = append(proofs, n.MerkleizeNode(addr>>(64-n.Depth), sibling))
	}

	return proofs
}

func (n *LargeRadixNode[C]) GenerateProof(addr uint64) [][32]byte {
	var proofs [][32]byte
	path := addressToRadixPath(addr, n.Depth, LargeRadixSize)

	if n.Children[path] == nil {
		proofs = zeroHashRange(0, 60-n.Depth-LargeRadixSize)
	} else {
		proofs = (*n.Children[path]).GenerateProof(addr)
	}

	for idx := path + 1<<LargeRadixSize; idx > 1; idx >>= 1 {
		sibling := idx ^ 1
		proofs = append(proofs, n.MerkleizeNode(addr>>(64-n.Depth), sibling))
	}
	return proofs
}

func (m *Memory) GenerateProof(addr uint64) [][32]byte {
	pageIndex := addr >> PageAddrSize

	if p, ok := m.pages[pageIndex]; ok {
		return p.GenerateProof(addr) // Generate proof from the page.
	} else {
		return zeroHashRange(0, 8) // Return zero hashes if the page does not exist.
	}
}

// MerkleizeNode computes the Merkle root hash for the node at the given generalized index.
// It recursively computes the hash of the subtree rooted at the given index.
// Note: The 'addr' parameter represents the partial address accumulated up to this node, not the full address. It represents the path taken in the trie to reach this node.
func (n *SmallRadixNode[C]) MerkleizeNode(addr, gindex uint64) [32]byte {
	depth := uint64(bits.Len64(gindex)) // Get the depth of the current gindex.

	if depth <= SmallRadixSize {
		hashBit := gindex & 15

		if (n.ChildExists & (1 << hashBit)) != 0 {
			if (n.HashValid & (1 << hashBit)) != 0 {
				// Return the cached hash if valid.
				return n.Hashes[gindex]
			} else {
				left := n.MerkleizeNode(addr, gindex<<1)
				right := n.MerkleizeNode(addr, (gindex<<1)|1)

				// Hash the pair and cache the result.
				r := HashPair(left, right)
				n.Hashes[gindex] = r
				n.HashValid |= 1 << hashBit
				return r
			}
		} else {
			// Return zero hash for non-existent child.
			return zeroHashes[64-5+1-(depth+n.Depth)]
		}
	}

	if depth > 5 {
		panic("gindex too deep")
	}

	childIndex := gindex - 1<<SmallRadixSize

	if n.Children[childIndex] == nil {
		// Return zero hash if child does not exist.
		return zeroHashes[64-5+1-(depth+n.Depth)]
	}

	// Update the partial address by appending the child index bits.
	// This accumulates the address as we traverse deeper into the trie.
	addr <<= SmallRadixSize
	addr |= childIndex
	return (*n.Children[childIndex]).MerkleizeNode(addr, 1)
}

func (n *LargeRadixNode[C]) MerkleizeNode(addr, gindex uint64) [32]byte {
	depth := uint64(bits.Len64(gindex))

	if depth <= LargeRadixSize {
		hashIndex := gindex >> 6
		hashBit := gindex & 63
		if (n.ChildExists[hashIndex] & (1 << hashBit)) != 0 {
			if (n.HashValid[hashIndex] & (1 << hashBit)) != 0 {
				return n.Hashes[gindex]
			} else {
				left := n.MerkleizeNode(addr, gindex<<1)
				right := n.MerkleizeNode(addr, (gindex<<1)|1)

				r := HashPair(left, right)
				n.Hashes[gindex] = r
				n.HashValid[hashIndex] |= 1 << hashBit
				return r
			}
		} else {
			return zeroHashes[64-5+1-(depth+n.Depth)]
		}
	}

	if depth > 16 {
		panic("gindex too deep")
	}

	childIndex := gindex - 1<<LargeRadixSize
	if n.Children[int(childIndex)] == nil {
		return zeroHashes[64-5+1-(depth+n.Depth)]
	}

	addr <<= LargeRadixSize
	addr |= childIndex
	return (*n.Children[childIndex]).MerkleizeNode(addr, 1)
}

func (m *Memory) MerkleizeNode(addr, gindex uint64) [32]byte {
	depth := uint64(bits.Len64(gindex))

	pageIndex := addr
	if p, ok := m.pages[pageIndex]; ok {
		return p.MerkleRoot()
	} else {
		return zeroHashes[64-5+1-(depth-1+52)]
	}
}

// MerkleRoot computes the Merkle root hash of the entire memory.
func (m *Memory) MerkleRoot() [32]byte {
	return (*m.radix).MerkleizeNode(0, 1)
}

// MerkleProof generates the Merkle proof for the specified address in memory.
func (m *Memory) MerkleProof(addr uint64) [ProofLen * 32]byte {
	proofs := m.radix.GenerateProof(addr)
	return encodeProofs(proofs)
}

// zeroHashRange returns a slice of zero hashes from start to end.
func zeroHashRange(start, end uint64) [][32]byte {
	proofs := make([][32]byte, end-start)
	if start == 0 {
		proofs[0] = zeroHashes[0]
		start++
	}
	for i := start; i < end; i++ {
		proofs[i] = zeroHashes[i-1]
	}
	return proofs
}

// encodeProofs encodes the list of proof hashes into a byte array.
func encodeProofs(proofs [][32]byte) [ProofLen * 32]byte {
	var out [ProofLen * 32]byte
	for i := 0; i < ProofLen; i++ {
		copy(out[i*32:(i+1)*32], proofs[i][:])
	}
	return out
}

// addressToRadixPath extracts a segment of bits from an address, starting from 'position' with 'count' bits.
// It returns the extracted bits as a uint64.
func addressToRadixPath(addr, position, count uint64) uint64 {
	// Calculate the total shift amount.
	totalShift := 64 - position - count

	// Shift the address to bring the desired bits to the LSB.
	addr >>= totalShift

	// Extract the desired bits using a mask.
	return addr & ((1 << count) - 1)
}

// addressToRadixPaths converts an address into a slice of radix path indices based on the branch factors.
func (m *Memory) addressToRadixPaths(addr uint64) []uint64 {
	path := make([]uint64, len(m.branchFactors))
	var position uint64

	for index, branchFactor := range m.branchFactors {
		path[index] = addressToRadixPath(addr, position, branchFactor)
		position += branchFactor
	}

	return path
}

// AllocPage allocates a new page at the specified page index in memory.
func (m *Memory) AllocPage(pageIndex uint64) *CachedPage {
	p := &CachedPage{Data: new(Page)}
	m.pages[pageIndex] = p

	addr := pageIndex << PageAddrSize
	branchPaths := m.addressToRadixPaths(addr)
	depth := uint64(0)

	// Build the radix trie path to the new page, creating nodes as necessary.
	radixLevel1 := m.radix
	depth += SmallRadixSize

	if (*radixLevel1).Children[branchPaths[0]] == nil {
		m.stats.Increment("1_alloc")
		node := &SmallRadixNode[L3]{Depth: depth}
		(*radixLevel1).Children[branchPaths[0]] = &node
	}
	radixLevel1.InvalidateNode(addr)

	radixLevel2 := (*radixLevel1).Children[branchPaths[0]]
	depth += SmallRadixSize
	if (*radixLevel2).Children[branchPaths[1]] == nil {
		m.stats.Increment("2_alloc")
		node := &SmallRadixNode[L4]{Depth: depth}
		(*radixLevel2).Children[branchPaths[1]] = &node
	}
	(*radixLevel2).InvalidateNode(addr)

	radixLevel3 := (*radixLevel2).Children[branchPaths[1]]
	depth += SmallRadixSize
	if (*radixLevel3).Children[branchPaths[2]] == nil {
		m.stats.Increment("3_alloc")
		node := &SmallRadixNode[L5]{Depth: depth}
		(*radixLevel3).Children[branchPaths[2]] = &node
	}
	(*radixLevel3).InvalidateNode(addr)

	radixLevel4 := (*radixLevel3).Children[branchPaths[2]]
	depth += SmallRadixSize
	if (*radixLevel4).Children[branchPaths[3]] == nil {
		m.stats.Increment("4_alloc")
		node := &SmallRadixNode[L6]{Depth: depth}
		(*radixLevel4).Children[branchPaths[3]] = &node
	}
	(*radixLevel4).InvalidateNode(addr)

	radixLevel5 := (*radixLevel4).Children[branchPaths[3]]
	depth += SmallRadixSize
	if (*radixLevel5).Children[branchPaths[4]] == nil {
		m.stats.Increment("5_alloc")
		node := &SmallRadixNode[L7]{Depth: depth}
		(*radixLevel5).Children[branchPaths[4]] = &node
	}
	(*radixLevel5).InvalidateNode(addr)

	radixLevel6 := (*radixLevel5).Children[branchPaths[4]]
	depth += SmallRadixSize
	if (*radixLevel6).Children[branchPaths[5]] == nil {
		m.stats.Increment("6_alloc")
		node := &SmallRadixNode[L8]{Depth: depth}
		(*radixLevel6).Children[branchPaths[5]] = &node
	}
	(*radixLevel6).InvalidateNode(addr)

	radixLevel7 := (*radixLevel6).Children[branchPaths[5]]
	depth += SmallRadixSize
	if (*radixLevel7).Children[branchPaths[6]] == nil {
		m.stats.Increment("7_alloc")
		node := &LargeRadixNode[L9]{Depth: depth}
		(*radixLevel7).Children[branchPaths[6]] = &node
	}
	(*radixLevel7).InvalidateNode(addr)

	radixLevel8 := (*radixLevel7).Children[branchPaths[6]]
	depth += LargeRadixSize
	if (*radixLevel8).Children[branchPaths[7]] == nil {
		m.stats.Increment("8_alloc")
		node := &LargeRadixNode[L10]{Depth: depth}
		(*radixLevel8).Children[branchPaths[7]] = &node
	}
	(*radixLevel8).InvalidateNode(addr)

	radixLevel9 := (*radixLevel8).Children[branchPaths[7]]
	depth += LargeRadixSize
	if (*radixLevel9).Children[branchPaths[8]] == nil {
		m.stats.Increment("9_alloc")
		node := &LargeRadixNode[L11]{Depth: depth}
		(*radixLevel9).Children[branchPaths[8]] = &node
	}
	(*radixLevel9).InvalidateNode(addr)
	m.stats.Increment("10_alloc")
	radixLevel10 := (*radixLevel9).Children[branchPaths[8]]
	(*radixLevel10).InvalidateNode(addr)
	(*radixLevel10).Children[branchPaths[9]] = &m

	m.InvalidateNode(addr)

	return p
}

// Invalidate invalidates the cache along the path from the specified address up to the root.
// It ensures that any cached hashes are recomputed when needed.
func (m *Memory) Invalidate(addr uint64) {
	// Find the page and invalidate the address within it.
	if p, ok := m.pageLookup(addr >> PageAddrSize); ok {
		prevValid := p.Ok[1]
		if !prevValid {
			// If the page was already invalid, the nodes up to the root are also invalid.
			return
		}
	} else {
		return
	}

	branchPaths := m.addressToRadixPaths(addr)

	currentLevel1 := m.radix
	currentLevel1.InvalidateNode(addr)
	m.stats.Increment("1_invalidate")
	radixLevel2 := (*m.radix).Children[branchPaths[0]]
	if radixLevel2 == nil {
		return
	}
	m.stats.Increment("2_invalidate")
	(*radixLevel2).InvalidateNode(addr)

	radixLevel3 := (*radixLevel2).Children[branchPaths[1]]
	if radixLevel3 == nil {
		return
	}
	m.stats.Increment("3_invalidate")
	(*radixLevel3).InvalidateNode(addr)

	radixLevel4 := (*radixLevel3).Children[branchPaths[2]]
	if radixLevel4 == nil {
		return
	}
	m.stats.Increment("4_invalidate")
	(*radixLevel4).InvalidateNode(addr)

	radixLevel5 := (*radixLevel4).Children[branchPaths[3]]
	if radixLevel5 == nil {
		return
	}
	m.stats.Increment("5_invalidate")
	(*radixLevel5).InvalidateNode(addr)

	radixLevel6 := (*radixLevel5).Children[branchPaths[4]]
	if radixLevel6 == nil {
		return
	}
	m.stats.Increment("6_invalidate")
	(*radixLevel6).InvalidateNode(addr)

	radixLevel7 := (*radixLevel6).Children[branchPaths[5]]
	if radixLevel7 == nil {
		return
	}
	m.stats.Increment("7_invalidate")
	(*radixLevel7).InvalidateNode(addr)

	radixLevel8 := (*radixLevel7).Children[branchPaths[6]]
	if radixLevel8 == nil {
		return
	}
	m.stats.Increment("8_invalidate")
	(*radixLevel8).InvalidateNode(addr)

	radixLevel9 := (*radixLevel8).Children[branchPaths[7]]
	if radixLevel9 == nil {
		return
	}
	m.stats.Increment("9_invalidate")
	(*radixLevel9).InvalidateNode(addr)

	radixLevel10 := (*radixLevel9).Children[branchPaths[8]]
	if radixLevel10 == nil {
		return
	}
	m.stats.Increment("10_invalidate")
	(*radixLevel10).InvalidateNode(addr)

	m.InvalidateNode(addr)
}

type Stats struct {
	counts map[string]int
	mutex  sync.Mutex // To ensure thread safety if accessing from multiple goroutines
}

func (s *Stats) Increment(key string) {
	s.mutex.Lock()
	s.counts[key]++
	s.mutex.Unlock()
}

func NewStats() *Stats {
	return &Stats{
		counts: make(map[string]int),
	}
}

func (s *Stats) WriteToCSV(filename string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Optionally, write a header
	if err := writer.Write([]string{"Key", "Count"}); err != nil {
		return err
	}

	for key, count := range s.counts {
		record := []string{key, strconv.Itoa(count)}
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return nil
}
