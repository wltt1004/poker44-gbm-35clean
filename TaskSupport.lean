module

public import Lean

@[expose] public section

open Lean Elab Meta Term

namespace FormalConjecturesVerifier

meta def constantType (source : TSyntax `str) : TermElabM Expr := do
  let sourceInfo ← getConstInfo source.getString.toName
  return sourceInfo.type

syntax (name := fcTypeOfName) "fcTypeOfName% " str : term

@[term_elab fcTypeOfName]
meta def elaborateTypeOfName : TermElab := fun stx _expectedType => do
  match stx with
  | `(fcTypeOfName% $source:str) => constantType source
  | _ => throwUnsupportedSyntax

end FormalConjecturesVerifier
