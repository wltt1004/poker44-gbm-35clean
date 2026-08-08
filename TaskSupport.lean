module

public import Lean

@[expose] public section

open Lean Elab Meta Term

namespace FormalConjecturesVerifier

partial meta def replaceAnswerWithTrue (expression : Expr) : MetaM Expr := do
  match expression with
  | .mdata metadata inner =>
      if metadata.contains `answer then
        return mkConst ``True
      return .mdata metadata (← replaceAnswerWithTrue inner)
  | .app function argument =>
      return .app (← replaceAnswerWithTrue function) (← replaceAnswerWithTrue argument)
  | .lam name domain body binderInfo =>
      return .lam name (← replaceAnswerWithTrue domain) (← replaceAnswerWithTrue body) binderInfo
  | .forallE name domain body binderInfo =>
      return .forallE name (← replaceAnswerWithTrue domain) (← replaceAnswerWithTrue body) binderInfo
  | .letE name type value body nondep =>
      return .letE name (← replaceAnswerWithTrue type) (← replaceAnswerWithTrue value)
        (← replaceAnswerWithTrue body) nondep
  | .proj typeName index sourceStructure =>
      return .proj typeName index (← replaceAnswerWithTrue sourceStructure)
  | other => return other

meta def constantType (source : TSyntax `str) : TermElabM Expr := do
  let sourceInfo ← getConstInfo source.getString.toName
  replaceAnswerWithTrue sourceInfo.type

syntax (name := fcTypeOfName) "fcTypeOfName% " str : term

@[term_elab fcTypeOfName]
meta def elaborateTypeOfName : TermElab := fun stx _expectedType => do
  match stx with
  | `(fcTypeOfName% $source:str) => constantType source
  | _ => throwUnsupportedSyntax

end FormalConjecturesVerifier
