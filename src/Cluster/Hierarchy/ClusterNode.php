<?php

declare(strict_types=1);

namespace Tessero\Cluster\Hierarchy;

/**
 * scipy.cluster.hierarchy.ClusterNode: a node in the binary tree built by to_tree() from a linkage matrix Z.
 * Each node carries its cluster id, its two children (null for a leaf), the merge distance and the number of
 * original observations below it. Pure PHP (no kernel); identical on both backends.
 */
final class ClusterNode
{
    public int $id;

    public ?ClusterNode $left;

    public ?ClusterNode $right;

    public float $dist;

    public int $count;

    public function __construct(int $id, ?ClusterNode $left = null, ?ClusterNode $right = null, float $dist = 0.0, ?int $count = null)
    {
        if ($left !== null && ($right === null) !== ($left === null)) {
            throw new \ValueError('Only full or proper binary trees are permitted. This node has one child.');
        }
        if ($count === null) {
            $count = $left !== null ? $left->count + $right->count : 1;
        }
        $this->id = $id;
        $this->left = $left;
        $this->right = $right;
        $this->dist = $dist;
        $this->count = $count;
    }

    /** The cluster id of this node (scipy ClusterNode.get_id). */
    public function getId(): int
    {
        return $this->id;
    }

    /** The number of original observations in the cluster rooted at this node (scipy get_count). */
    public function getCount(): int
    {
        return $this->count;
    }

    /** The left child, or null for a leaf (scipy get_left). */
    public function getLeft(): ?ClusterNode
    {
        return $this->left;
    }

    /** The right child, or null for a leaf (scipy get_right). */
    public function getRight(): ?ClusterNode
    {
        return $this->right;
    }

    /** True when this node is a leaf (an original observation). scipy is_leaf. */
    public function isLeaf(): bool
    {
        return $this->left === null;
    }

    /**
     * Left-first pre-order traversal returning a leaf value per original observation (scipy pre_order).
     * The default $func returns each leaf's id; pass a callable to map each leaf ClusterNode to a value.
     * Iterative (matches scipy's explicit-stack implementation and avoids recursion limits).
     *
     * @param callable(ClusterNode):mixed|null $func
     *
     * @return list<mixed>
     */
    public function preOrder(?callable $func = null): array
    {
        $func ??= static fn (ClusterNode $x): int => $x->id;
        $n = 2 * $this->count - 1;
        $curNode = array_fill(0, $n, null);
        $lvisited = [];
        $rvisited = [];
        $curNode[0] = $this;
        $k = 0;
        $preorder = [];
        while ($k >= 0) {
            $nd = $curNode[$k];
            $ndid = $nd->id;
            if ($nd->isLeaf()) {
                $preorder[] = $func($nd);
                $k--;
            } else {
                if (! isset($lvisited[$ndid])) {
                    $curNode[$k + 1] = $nd->left;
                    $lvisited[$ndid] = true;
                    $k++;
                } elseif (! isset($rvisited[$ndid])) {
                    $curNode[$k + 1] = $nd->right;
                    $rvisited[$ndid] = true;
                    $k++;
                } else {
                    $k--;
                }
            }
        }

        return $preorder;
    }
}
