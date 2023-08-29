
human_labels = ("Duration (in seconds),RecordedDate,Prolific ID,Investment_Chain,Sex_Chain,Alien_Chain,"+
    "Sex_CCG,Alien_CCG,Investment_CCG,Sex_CCP,Alien_CCP,Investment_CCP,Investment_ControlP,Alien_ControlP,"+
    "Sex_ControlP,Alien_ControlG,Sex_ControlG,Investment_ControlG,Att_Check_18,Gender,Age,Backgroupn").split(",")

query_variants = ["alien", "econ", "sex"]

human_col_mapping = {
    "control_sex": "Sex_ControlG",
    "control_econ": "Investment_ControlG",
    "control_alien": "Alien_ControlG",
    "chain_econ": "Investment_Chain",
    "chain_sex": "Sex_Chain",
    "chain_alien": "Alien_Chain",
    "cc_sex": "Sex_CCG",
    "cc_econ": "Investment_CCG",
    "cc_alien": "Alien_CCG",
    #"ccp_sex": "Sex_CCP", # FIXME these do not exists yet
    #"ccp_econ": "Investment_CCP",
    #"ccp_alien": "Alien_CCP"
    # "Investment_ControlP","Alien_ControlP", "Sex_ControlP"
}
