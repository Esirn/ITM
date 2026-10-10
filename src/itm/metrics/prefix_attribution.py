"""Two-ordering algebra of saved prefix predictions; not unique causal attribution."""
import numpy as np

from itm.metrics.training_sufficiency import VIEWS,_features,_bound


def two_component_delta(prediction,intermediate,reference,target,valid=None,*,direct_is_first=True):
    p,g,mask=_features(prediction,target,valid)
    i,_,_=_features(intermediate,target,mask);r,_,_=_features(reference,target,mask)
    p,i,r,g=[a[mask] for a in (p,i,r,g)]
    first,second=p-i,i-r
    d,s=(first,second) if direct_is_first else (second,first)
    total=p-r
    dc,sc,rc=[a-a.mean(0) for a in (d,s,r)];pc=p-p.mean(0)
    result={}
    for name,ids in VIEWS.items():
        mean=lambda a:float(np.mean(a[:,ids]))
        old,new=mean((r-g)**2),mean((p-g)**2)
        de,se,cross=mean(d*d),mean(s*s),2*mean(d*s)
        ed,es=2*mean((r-g)*d),2*mean((r-g)*s)
        energy=mean(total*total);change=new-old
        tde,tse,tcross=mean(dc*dc),mean(sc*sc),2*mean(dc*sc)
        old_t,new_t=mean(rc*rc),mean(pc*pc)
        td,ts=2*mean(rc*dc),2*mean(rc*sc)
        delta_t=new_t-old_t
        err=max(abs(energy-de-se-cross),abs(change-ed-es-de-se-cross),
                abs(mean((dc+sc)**2)-tde-tse-tcross),abs(delta_t-td-ts-tde-tse-tcross))
        bound=_bound(old,new,de,se,cross,ed,es,energy,old_t,new_t,tde,tse,tcross,td,ts)
        if err>bound:raise ArithmeticError('Two-component saved-prediction identity failed')
        result[name]=dict(reference_mse=old,prediction_mse=new,delta_mse=change,
            direct_energy=de,state_energy=se,twice_direct_state_covariance=cross,
            direct_error_cross=ed,state_error_cross=es,total_difference_energy=energy,
            direct_rms=float(np.sqrt(de)),state_rms=float(np.sqrt(se)),
            direct_centered_energy=tde,state_centered_energy=tse,twice_centered_covariance=tcross,
            reference_temporal_energy=old_t,prediction_temporal_energy=new_t,delta_temporal_energy=delta_t,
            twice_reference_direct_temporal_covariance=td,twice_reference_state_temporal_covariance=ts,
            identity_error=err,identity_bound=bound)
    result['balanced']={k:.5*result['root_ric67'][k]+.5*result['remaining196'][k]
        for k in ('reference_mse','prediction_mse','delta_mse','direct_energy','state_energy',
                  'twice_direct_state_covariance','direct_error_cross','state_error_cross',
                  'total_difference_energy','direct_centered_energy','state_centered_energy',
                  'twice_centered_covariance','delta_temporal_energy')}
    return result
