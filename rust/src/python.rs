use pyo3::exceptions::PySyntaxError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use rustpython_parser::{Parse, ast};

#[pyclass]
pub struct Ast(ast::Stmt);

#[pymethods]
impl Ast {
    #[new]
    fn new(code: &str, source_path: &str, 
                offset: u32) -> PyResult<Ast> {
        Ok(Ast(ast::Stmt::parse_starts_at(code, source_path, ast::TextSize::new(offset))
            .map_err(|_| PySyntaxError::new_err("Syntax error"))?))
    }

    fn __repr__(&self) -> String {
        return format!("{:?}", self.0);
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Expr {

}

/*
 * An "unbound" expresssion which may contain free variables.
 * The capture() method returns an Expr with the free variables bound.
 * This must be done after all free variables have been defined. i.e.
 * 
 * @func
 * def foo(): return bar()
 * 
 * foo._capture() <--- fails! bar is not defined.
 * 
 * @func
 * def bar(): return foo()
 * 
 * foo._capture() <--- succeeds. foo and bar are defined 
 *                    and can be mutually recursively captured.
 */
#[pyclass]
#[allow(dead_code)]
pub struct Function {
    expr: Expr, 
    scope: Py<PyDict>,
}

#[pymethods]
impl Function {
    #[new]
    fn new(expr: Expr, scope: &PyDict) -> PyResult<Self> {
        Ok(Function {expr, scope: scope.into()})
    }

    fn _capture(&self) -> PyResult<Expr> {
        Ok(Expr {})
    }
}

/*
 * A top-level expression which is not bound to a function.
 */
#[pyclass]
pub struct Entrypoint {

}

#[pyclass]
pub struct DeviceType {

}

#[pyclass]
pub struct Type {

}
/// A Python module implemented in Rust.
#[pymodule]
pub fn _stanza(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Ast>()?;
    m.add_class::<Type>()?;
    m.add_class::<Expr>()?;
    m.add_class::<Function>()?;
    Ok(())
}