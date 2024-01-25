use std::collections::HashMap;

pub enum PrimitiveType {
    Bool, 
    U8, U16, U32, U64,
    I8, I16, I32, I64,
    F32, F64,
    String,
}

pub enum Primitive {
    Bool(bool),
}

pub enum Value {
    Primitive(Primitive),
    Function(Entrypoint)
}

pub enum Type {
    Ident(String),
    PrimitiveType(PrimitiveType),
    List(Box<Type>), Tuple(Vec<Type>),
    Dict(HashMap<String, Type>),
    Array(Box<Type>, Vec<usize>),
    Function, Struct
}

pub enum Expr {

}