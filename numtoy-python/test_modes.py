import numtoy as nt
engine = nt.Engine()
A = nt.Expr.var(engine, 'A', [1.0, 2.0, 3.0])
B = nt.Expr.var(engine, 'B', [4.0, 5.0, 6.0])

print('default:', (A+B).execute(engine).unpack(engine))

with nt.mode('eager'):
    print('eager:', (A+B).execute(engine).unpack(engine))
    
with nt.mode('hyper'):
    print('hyper:', (A+B).execute(engine).unpack(engine))
