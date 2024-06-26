use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use logos::Logos;
use ordered_float::NotNan;

pub use super::{Type, Constant};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum Tree {
    Erase,
    Constant(Constant),
    // For wiring
    Var(String),
    // Reference to a net
    Ref(String),
    // Builtin operator
    Operator(String, Vec<Tree>),
    Con(Option<Type>, Vec<Tree>),
    Dup(Vec<Tree>),
    // switch equality cases
    // Switch(Vec<(Constant, Tree)>),
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Redex {
    pub lhs: Tree,
    pub rhs: Tree
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Net {
  pub root: Tree,
  pub redexs: Vec<Redex>,
}


#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Book {
    pub defs: BTreeMap<String, Net>,
}

impl Display for Tree {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            Tree::Constant(c) => write!(f, "{c}"),
            Tree::Var(v) => write!(f, "{v}"),
            Tree::Ref(r) => write!(f, "@{r}"),
            Tree::Erase => write!(f, "*"),
            Tree::Operator(op, args) => {
                write!(f, "${op}")?;
                if !args.is_empty() {
                    write!(f, "[")?;
                    let mut first = true;
                    for arg in args {
                        if first { write!(f, "{}", arg)?; } 
                        else { write!(f, ", {}", arg)?; }
                        first = false;
                    }
                    write!(f, "]")?;
                }
                Ok(())
            },
            Tree::Con(ty, args) => {
                if let Some(ty) = ty {
                    write!(f, "{ty}")?;
                }
                write!(f, "(")?;
                if args.is_empty() { write!(f, ",")?; }
                let mut first = true;
                for arg in args {
                    if first { write!(f, "{}", arg)?; } 
                    else { write!(f, ", {}", arg)?; }
                    first = false;
                }
                write!(f, ")")
            },
            Tree::Dup(args) => {
                write!(f, "{{")?;
                let mut first = true;
                for arg in args {
                    if first { write!(f, "{}", arg)?; } 
                    else { write!(f, " {}", arg)?; }
                    first = false;
                }
                write!(f, "}}")
            }
        }
    }
}

impl Display for Redex {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{lhs} ~ {rhs}", lhs = self.lhs, rhs = self.rhs)
    }
}

impl Display for Net {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{root}", root = self.root)?;
        for redex in &self.redexs {
            write!(f, " &\n\t {redex}", redex = redex)?;
        }
        Ok(())
    }
}

impl Display for Book {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        for (name, net) in &self.defs {
            write!(f, "@{name} = {net}\n")?;
        }
        Ok(())
    }
}

// The token for the core language
#[derive(Logos, Debug, PartialEq, Eq, Clone)]
pub enum Token<'src> {
    #[regex(r"[ \t\n\f]*", logos::skip)]
    Whitespace,
    #[regex(r"/\*([^\*]*\*[^/])*[^\*]*\*/", logos::skip)]
    BlockComment,
    #[regex(r"//[^\n]*", logos::skip)]
    LineComment,

    #[regex(r"[a-z][a-zA-Z_0-9]*")]
    Identifier(&'src str),

    #[regex(r"[A-Z][a-zA-Z_0-9]*(#[A-Z0-9]*)?", |x| {
        let s = x.slice();
        (s, Some(s))
    })]
    Type((&'src str,Option<&'src str>)),

    #[regex(r"@[a-zA-Z][a-zA-Z_0-9]*", |x| {let s = x.slice(); &s[1..]})]
    Reference(&'src str),
    #[regex(r"\$[a-zA-Z][a-zA-Z_0-9]*", |x| {let s = x.slice(); &s[1..]})]
    Operator(&'src str),

    #[token("*")]
    Star,
    #[token("=")]
    Equals,
    #[token(",")]
    Comma,
    #[token("~")]
    Tilde,
    #[token("&")]
    Ampersand,

    #[token("(")] LParen, #[token(")")] RParen,
    #[token("{")] LBrace, #[token("}")] RBrace,
    #[token("[")] LBracket, #[token("]")] RBracket,

    #[regex(r"[0-9]+", |x| x.slice().parse())]
    Integer(u64),
    #[regex("\"[^\"]*\"", |x| {let s = x.slice(); &s[1..s.len() - 1]})]
    String(&'src str),
    #[regex(r"[0-9]+\.[0-9]+", |x| x.slice().parse())]
    Float(NotNan<f64>),
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[error]
    Error
}

use logos::Lexer as LogosLexer;

pub struct Lexer<'src> {
    logos_lex : LogosLexer<'src, Token<'src>>,
}

impl<'src> Lexer<'src> {
    pub fn new(src: &'src str) -> Lexer {
        Lexer { 
            logos_lex: Token::lexer(src)
        }
    }
}

impl<'src> Iterator for Lexer<'src> {
    type Item = Token<'src>;
    
    fn next(&mut self) -> Option<Self::Item> {
        self.logos_lex.next()
    }
}