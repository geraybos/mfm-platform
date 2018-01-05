import sys
from opt_portfolio_prod import opt_portfolio_prod
# 500套保

date = sys.argv[1]
json_dir = sys.argv[2]
output_dir = sys.argv[3]

opt = opt_portfolio_prod(date=date, read_json=json_dir, output_dir=output_dir,
                         benchmark='zz500', stock_pool='zz500')
opt.execute_opt()