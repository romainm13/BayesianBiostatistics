import sys; sys.path.append("../")
from bayestrees import Model, create_theta, affiche_N, affiche_theta_no

modele1 = Model(3)
modele1.tree = ((3,),(),(),(1,2))
(modele1.theta_o, modele1.theta_no) = create_theta(modele1)

affiche_N(modele1,'les_Nij = 0,5.pdf')
affiche_theta_no(modele1,'les_theta.pdf')
