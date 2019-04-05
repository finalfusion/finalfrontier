use std::mem;

use conllx::graph::{DepGraph, DepTriple};

use crate::DepembedsConfig;

/// Trait to provide iterators over the path in a tree from `start` to the root.
pub trait PathIter {
    fn path_iter(&self, start: usize) -> PathIterator;
}

impl<'a> PathIter for DepGraph<'a> {
    fn path_iter(&self, start: usize) -> PathIterator {
        PathIterator {
            graph: self,
            current: start,
        }
    }
}

/// Iterator over the path from the given start node to the root node.
///
/// The path does not include the start node itself.
pub struct PathIterator<'a, 'b> {
    current: usize,
    graph: &'a DepGraph<'b>,
}

impl<'a, 'b> Iterator for PathIterator<'a, 'b> {
    type Item = DepTriple<&'b str>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(triple) = self.graph.head(self.current) {
            self.current = triple.head();
            Some(triple)
        } else {
            None
        }
    }
}

/// Enum for different types of dependencies. Typed through direction, depth, attached form and label.
#[derive(Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum Dependency {
    /// Typed dependency through Direction (`Regular` and `Inverse`), depth, relation label and form.
    Typed {
        direction: DependencyDirection,
        depth: usize,
        dep_label: String,
        form: String,
    },
    /// Untyped dependency just denoting that there exists any kind of relation.
    Untyped(String),
}

// Constructors for convenience
impl Dependency {
    fn regular<S, T>(depth: usize, dep_label: S, form: T) -> Self
    where
        S: Into<String>,
        T: Into<String>,
    {
        Dependency::Typed {
            direction: DependencyDirection::Regular,
            depth,
            dep_label: dep_label.into(),
            form: form.into(),
        }
    }
    fn inverse<S, T>(depth: usize, dep_label: S, form: T) -> Self
    where
        S: Into<String>,
        T: Into<String>,
    {
        Dependency::Typed {
            direction: DependencyDirection::Inverse,
            depth,
            dep_label: dep_label.into(),
            form: form.into(),
        }
    }
}

/// Enum to denote the direction of a dependency relation.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum DependencyDirection {
    /// Inverse relation: relation seen from a dependent to its head.
    Inverse,
    /// Regular relation: relation seen from a head to a dependent.
    Regular,
}

/// Struct to iterate over the dependencies in a `conllx::DepGraph`.
///
/// Provides tuples with form `(focus_idx, Dependency)` where focus_idx is the index of the
/// focus token for the Dependency context.
pub struct DependencyIterator<'a> {
    max_depth: usize,
    cur: usize,
    depth: usize,
    graph: &'a DepGraph<'a>,
    path_iter: PathIterator<'a, 'a>,
    buffer: Option<(usize, Dependency)>,
}

impl<'a> DependencyIterator<'a> {
    /// Constructs a new `DependencyIterator` which returns up to `max_depth`-order dependencies.
    ///
    /// If `max_depth == 0`, all contexts are extracted.
    pub fn new(graph: &'a DepGraph<'a>, max_depth: usize) -> Self {
        DependencyIterator {
            max_depth,
            cur: 1,
            depth: 0,
            buffer: None,
            graph,
            path_iter: graph.path_iter(1),
        }
    }

    /// Construct a `DependencyIterator` and apply parameters given in `config`.
    pub fn new_from_config(
        graph: &'a DepGraph<'a>,
        config: DepembedsConfig,
    ) -> Box<Iterator<Item = (usize, Dependency)> + 'a> {
        let iter = DependencyIterator::new(graph, config.depth as usize);

        match (config.normalize, config.untyped, config.use_root) {
            (false, false, false) => Box::new(iter.filter_root()),
            (false, false, true) => Box::new(iter),
            (false, true, false) => Box::new(iter.untyped().filter_root()),
            (false, true, true) => Box::new(iter.untyped()),
            (true, false, false) => Box::new(iter.normalized().filter_root()),
            (true, false, true) => Box::new(iter.normalized()),
            (true, true, false) => Box::new(iter.normalized().untyped().filter_root()),
            (true, true, true) => Box::new(iter.normalized().untyped()),
        }
    }

    /// Constructs a `Dependency` context with `DependencyDirection::Inverse` for the token at index
    /// `self.cur`.
    fn inverse_context(&self, triple: &DepTriple<&str>, depth: usize) -> Dependency {
        let rel = triple.relation().unwrap_or_default();
        if let Some(token) = self.graph[triple.head()].token() {
            Dependency::inverse(depth, rel, token.form())
        } else {
            Dependency::inverse(depth, rel, "<root>")
        }
    }

    /// Constructs a `Dependency` context with `DependencyDirection::Regular` for a head-token.
    fn regular_context(&self, triple: &DepTriple<&str>, depth: usize) -> Dependency {
        let rel = triple.relation().unwrap_or_default();
        if let Some(token) = self.graph[triple.dependent()].token() {
            Dependency::regular(depth, rel, token.form())
        } else {
            Dependency::regular(depth, rel, "<root>")
        }
    }
}

impl<'a> Iterator for DependencyIterator<'a> {
    type Item = (usize, Dependency);

    fn next(&mut self) -> Option<(usize, Dependency)> {
        // possibly return stored regular dependency
        if self.buffer.is_some() {
            return self.buffer.take();
        }

        // while loop moves through sentence
        while self.cur < self.graph.len() {
            // climb up the tree one step per next() call
            if let Some(triple) = self.path_iter.next() {
                if (self.depth == self.max_depth) && (self.max_depth != 0) {
                    continue;
                }
                self.depth += 1;

                // guard against int underflow since root idx is 0
                if triple.head() != 0 {
                    // unwrap is safe here because self.path_iter.next() has to make the same check.
                    let cur_triple = self.graph.head(self.cur).unwrap();
                    // regular dependency is context for the head of the triple but is typed through
                    // the token at self.cur and its incoming edge
                    self.buffer = Some((
                        triple.head() - 1,
                        self.regular_context(&cur_triple, self.depth),
                    ));
                }
                // inverse dependencies are contexts of self.cur and typed through the head of the
                // triple and the outgoing relation of that head
                return Some((self.cur - 1, self.inverse_context(&triple, self.depth)));
            }
            self.cur += 1;
            self.depth = 0;
            self.path_iter = self.graph.path_iter(self.cur);
        }
        None
    }
}

/// Trait offering adapters for `DependencyIterator`.
pub trait DepIter: Sized {
    /// Normalizes the `form` in `Dependency` through lower-casing.
    fn normalized(self) -> Normalized<Self>;
    /// Maps `Dependency::Typed` to `Dependency::Untyped`.
    fn untyped(self) -> Untyped<Self>;
    /// Removes `Dependency`s with `form == "<root>"`
    fn filter_root(self) -> FilterRoot<Self>;
}

impl<I> DepIter for I
where
    I: Iterator<Item = (usize, Dependency)>,
{
    fn normalized(self) -> Normalized<I> {
        Normalized { inner: self }
    }
    fn untyped(self) -> Untyped<I> {
        Untyped { inner: self }
    }
    fn filter_root(self) -> FilterRoot<I> {
        FilterRoot { inner: self }
    }
}

/// Adapter for iterators over `(usize, Dependency)` to filter `Dependency`s with `form == "<root>"`
pub struct FilterRoot<I> {
    inner: I,
}

impl<I> Iterator for FilterRoot<I>
where
    I: Iterator<Item = (usize, Dependency)>,
{
    type Item = (usize, Dependency);

    fn next(&mut self) -> Option<(usize, Dependency)> {
        while let Some(tuple) = self.inner.next() {
            match tuple.1 {
                Dependency::Typed { ref form, .. } => {
                    if form == "<root>" {
                        continue;
                    }
                }
                Dependency::Untyped(ref form) => {
                    if form == "<root>" {
                        continue;
                    }
                }
            }
            return Some(tuple);
        }
        None
    }
}

/// Adapter for iterators over `(usize, Dependency)` to normalize the `form` in the `Dependency`.
pub struct Normalized<I> {
    inner: I,
}

impl<I> Iterator for Normalized<I>
where
    I: Iterator<Item = (usize, Dependency)>,
{
    type Item = (usize, Dependency);

    fn next(&mut self) -> Option<(usize, Dependency)> {
        self.inner
            .next()
            .map(|mut tuple| {
                match tuple.1 {
                    Dependency::Untyped(ref mut form) => {
                        let normalized = form.to_lowercase();
                        mem::replace(form, normalized);
                    }
                    Dependency::Typed { ref mut form, .. } => {
                        let normalized = form.to_lowercase();
                        mem::replace(form, normalized);
                    }
                }
                tuple
            })
            .take()
    }
}

/// Adapter for iterators over `(usize, Dependency)` to map `Dependency::Typed` to
/// `Dependency::Untyped`.
///
/// The adapter takes the `form` from the input `Dependency` and wraps it in `Dependency::Untyped`.
pub struct Untyped<I> {
    inner: I,
}

impl<I> Iterator for Untyped<I>
where
    I: Iterator<Item = (usize, Dependency)>,
{
    type Item = (usize, Dependency);

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        self.inner
            .next()
            .map(|mut tuple| {
                if let Dependency::Typed { form, .. } = tuple.1 {
                    tuple.1 = Dependency::Untyped(form);
                }
                tuple
            })
            .take()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::deps::{DepIter, Dependency, Dependency::Untyped, DependencyIterator, PathIter};

    use conllx::graph::Node;
    use conllx::io::{ReadSentence, Reader};

    static DEP: &[u8; 143] = b"1	Er	a	_	_	_	2	SUBJ	_	_\n\
    2	geht	b	_	_	_	0	ROOT	_	_\n\
    3	ins	c	_	_	_	2	PP	_	_\n\
    4	Kino	d	_	_	_	3	PN	_	_\n\
    5	root2	e	_	_	_	0	ROOT	_	_\n\
    6	dep	f	_	_	_	5	DEP";

    #[test]
    fn paths() {
        let c = Cursor::new(DEP.to_vec());
        let mut reader = Reader::new(c);
        let v = vec![
            vec!["geht".to_string(), "".to_string()], // ER
            vec!["".to_string()],                     // GEHT
            vec!["geht".to_string(), "".to_string()], // INS
            vec!["ins".to_string(), "geht".to_string(), "".to_string()], // KINO
            vec!["".to_string()],                     //root2
            vec!["root2".to_string(), "".to_string()], //dep
        ];
        let sentence = reader.read_sentence().unwrap().unwrap();

        let g = sentence.dep_graph();
        assert_eq!(g.len() - 1, v.len());
        for (target, node) in v.into_iter().zip(1..g.len()) {
            let path = g.path_iter(node);
            assert_eq!(
                path.map(|triple| triple.head())
                    .map(|head| match &g[head] {
                        Node::Token(token) => token.form().to_owned(),
                        Node::Root => "".to_owned(),
                    })
                    .collect::<Vec<_>>(),
                target
            );
        }
    }

    #[test]
    pub fn dep_iter_typed_with_root_depth1() {
        let c = Cursor::new(DEP.to_vec());
        let mut reader = Reader::new(c);

        let sentence = reader.read_sentence().unwrap().unwrap();
        let target_deps = vec![
            (0, Dependency::inverse(1, "SUBJ", "geht")),
            (1, Dependency::regular(1, "SUBJ", "Er")), // er
            (1, Dependency::inverse(1, "ROOT", "<root>")), // geht
            (2, Dependency::inverse(1, "PP", "geht")),
            (1, Dependency::regular(1, "PP", "ins")), // ins
            (3, Dependency::inverse(1, "PN", "ins")),
            (2, Dependency::regular(1, "PN", "Kino")), // kino
            (4, Dependency::inverse(1, "ROOT", "<root>")), // root2
            (5, Dependency::inverse(1, "DEP", "root2")),
            (4, Dependency::regular(1, "DEP", "dep")), // dep
        ];
        let deps = DependencyIterator::new(&sentence.dep_graph(), 1).collect::<Vec<_>>();
        assert_eq!(deps.len(), target_deps.len());
        for (dep, target_dep) in deps.into_iter().zip(target_deps) {
            assert_eq!(dep, target_dep);
        }
    }

    #[test]
    pub fn dep_iter_typed_no_root_depth1() {
        let c = Cursor::new(DEP.to_vec());

        let target_deps = vec![
            (0, Dependency::inverse(1, "SUBJ", "geht")),
            (1, Dependency::regular(1, "SUBJ", "Er")),
            (2, Dependency::inverse(1, "PP", "geht")),
            (1, Dependency::regular(1, "PP", "ins")),
            (3, Dependency::inverse(1, "PN", "ins")),
            (2, Dependency::regular(1, "PN", "Kino")),
            (5, Dependency::inverse(1, "DEP", "root2")),
            (4, Dependency::regular(1, "DEP", "dep")),
        ];
        let mut reader = Reader::new(c);
        let sentence = reader.read_sentence().unwrap().unwrap();
        let deps = DependencyIterator::new(&sentence.dep_graph(), 1)
            .filter_root()
            .collect::<Vec<_>>();
        assert_eq!(deps.len(), target_deps.len());
        for (dep, target_dep) in deps.into_iter().zip(target_deps) {
            assert_eq!(dep, target_dep);
        }
    }

    #[test]
    pub fn dep_iter_normalized_typed_no_root_depth2() {
        let target_deps = vec![
            (0, Dependency::inverse(1, "SUBJ", "geht")),
            (1, Dependency::regular(1, "SUBJ", "er")),
            (2, Dependency::inverse(1, "PP", "geht")),
            (1, Dependency::regular(1, "PP", "ins")),
            (3, Dependency::inverse(1, "PN", "ins")),
            (2, Dependency::regular(1, "PN", "kino")),
            (3, Dependency::inverse(2, "PP", "geht")),
            (1, Dependency::regular(2, "PN", "kino")),
            (5, Dependency::inverse(1, "DEP", "root2")),
            (4, Dependency::regular(1, "DEP", "dep")),
        ];

        let c = Cursor::new(DEP.to_vec());
        let mut reader = Reader::new(c);

        let sentence = reader.read_sentence().unwrap().unwrap();

        let deps = DependencyIterator::new(&sentence.dep_graph(), 2)
            .normalized()
            .filter_root()
            .collect::<Vec<_>>();
        assert_eq!(deps.len(), target_deps.len());
        for (dep, target_dep) in deps.into_iter().zip(target_deps) {
            assert_eq!(dep, target_dep);
        }
    }

    #[test]
    pub fn dep_iter_untyped_with_root_depth2() {
        let target_deps = vec![
            // reachable from "er"
            (0, Untyped("geht".to_string())),
            (1, Untyped("er".to_string())),
            (0, Untyped("<root>".to_string())),
            // reachable from "geht"
            (1, Untyped("<root>".to_string())),
            // reachable from "ins"
            (2, Untyped("geht".to_string())),
            (1, Untyped("ins".to_string())),
            (2, Untyped("<root>".to_string())),
            // reachable from "Kino"
            (3, Untyped("ins".to_string())),
            (2, Untyped("kino".to_string())),
            (3, Untyped("geht".to_string())),
            (1, Untyped("kino".to_string())),
            // reachable from "root2"
            (4, Untyped("<root>".to_string())),
            // reachable from "dep"
            (5, Untyped("root2".to_string())),
            (4, Untyped("dep".to_string())),
            (5, Untyped("<root>".to_string())),
        ];

        let c = Cursor::new(DEP.to_vec());
        let mut reader = Reader::new(c);

        let sentence = reader.read_sentence().unwrap().unwrap();
        let deps = DependencyIterator::new(&sentence.dep_graph(), 2)
            .normalized()
            .untyped()
            .collect::<Vec<_>>();
        assert_eq!(deps.len(), target_deps.len());
        for (dep, target_dep) in deps.into_iter().zip(target_deps) {
            assert_eq!(dep, target_dep);
        }
    }

    #[test]
    pub fn dep_iter_typed_with_root_depth2() {
        let target_deps = vec![
            (0, Dependency::inverse(1, "SUBJ", "geht")),
            (1, Dependency::regular(1, "SUBJ", "er")),
            (0, Dependency::inverse(2, "ROOT", "<root>")),
            (1, Dependency::inverse(1, "ROOT", "<root>")),
            (2, Dependency::inverse(1, "PP", "geht")),
            (1, Dependency::regular(1, "PP", "ins")),
            (2, Dependency::inverse(2, "ROOT", "<root>")),
            (3, Dependency::inverse(1, "PN", "ins")),
            (2, Dependency::regular(1, "PN", "kino")),
            (3, Dependency::inverse(2, "PP", "geht")),
            (1, Dependency::regular(2, "PN", "kino")),
            (4, Dependency::inverse(1, "ROOT", "<root>")),
            (5, Dependency::inverse(1, "DEP", "root2")),
            (4, Dependency::regular(1, "DEP", "dep")),
            (5, Dependency::inverse(2, "ROOT", "<root>")),
        ];

        let c = Cursor::new(DEP.to_vec());
        let mut reader = Reader::new(c);

        let sentence = reader.read_sentence().unwrap().unwrap();

        let deps = DependencyIterator::new(&sentence.dep_graph(), 2)
            .normalized()
            .collect::<Vec<_>>();
        assert_eq!(deps.len(), target_deps.len());
        for (dep, target_dep) in deps.into_iter().zip(target_deps) {
            assert_eq!(dep, target_dep);
        }
    }
}
